# %% pre-extracting f0
import json
import os
import sys
import math

import librosa.core
from tqdm import tqdm
import torch
import pyworld as pw
import numpy as np
import argparse
from multiprocessing import Pool

sys.path.append('/mnt/sdb/liruiqi/SingingDictation')
from modules.pe.rmvpe import RMVPE
from data_gen.tts.base_binarizer import BaseBinarizer
from utils.commons.hparams import hparams, set_hparams
from utils.commons.dataset_utils import batch_by_size, pad_or_cut_xd
from utils.audio.pitch_utils import hz_to_midi, f0_to_coarse, resample_align_curve

class F0Extractor:
    def __init__(self, save_dir):
        self.processed_data_dir = hparams['processed_data_dir']
        self.binarization_args = hparams['binarization_args']
        self.items = {}
        self.item_names = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir

    def load_meta_data(self):
        metafile_path = hparams.get('metafile_path', f"{self.processed_data_dir}/metadata.json")
        if ',' in metafile_path:
            metafile_paths = metafile_path.split(',')
            ds_names = hparams.get('ds_names', ','.join([str(i) for i in range(len(metafile_paths))])).split(',')
        else:
            metafile_paths = [metafile_path]
            ds_names = [hparams.get('ds_names', '0')]
        for idx, metafile_path in enumerate(metafile_paths):
            items_list = json.load(open(metafile_path))
            for r in tqdm(items_list, desc=f'| Loading meta data for dataset {ds_names[idx]}.'):
                item_name = r['item_name']
                if item_name in self.items:
                    print(f'warning: item name {item_name} duplicated')
                self.items[item_name] = r
                self.item_names.append(item_name)
                self.items[item_name]['ds_name'] = ds_names[idx]

    @staticmethod
    def generate(items, hparams, binarization_args, device=None, task_id=-1):
        f0_dict = {}
        pe = hparams.get('pe', 'pw')
        if pe == 'rmvpe':
            rmvpe = RMVPE(hparams['pe_ckpt'], device=device)
        for item_name in tqdm(items.keys(), total=len(items),
                              desc='Extracting' + (f' {task_id}' if task_id >= 0 else ''), position=task_id):
            item = items[item_name]
            wav_fn = item['wav_fn']
            wav, mel = BaseBinarizer.process_audio(wav_fn, item, binarization_args)

            f0 = None
            if pe == 'rmvpe':
                wav, length = item['wav'], item['mel'].shape[0]
                with torch.no_grad():
                    f0, uv = rmvpe.get_pitch(
                        wav, sample_rate=hparams['audio_sample_rate'],
                        hop_size=hparams['hop_size'],
                        length=(wav.shape[0] + hparams['hop_size'] - 1) // hparams['hop_size'],
                        fmax=hparams['f0_max'],
                        fmin=hparams['f0_min']
                    )
            elif pe == 'pw':
                f0, _ = pw.harvest(wav.astype(np.double), hparams['audio_sample_rate'],
                                   frame_period=hparams['hop_size'] * 1000 / hparams['audio_sample_rate'])
                delta_l = length - len(f0)
                if delta_l < 0:
                    f0 = f0[:length]
                elif delta_l > 0:
                    f0 = np.concatenate((f0, np.full(delta_l, fill_value=f0[-1])), axis=0)

            f0_dict[item_name] = f0

        return f0_dict

    @staticmethod
    def generate_batch(items, hparams, binarization_args, device=None, task_id=-1, bsz=1, max_tokens=100000):
        f0_dict = {}
        pe = hparams.get('pe', 'pw')
        rmvpe = RMVPE(hparams['pe_ckpt'], device=device)

        item_names = list(items.keys())
        id_and_sizes = []
        for item_name in item_names:
            total_durs = np.sum(items[item_name]['word_durs'])
            total_frames = math.ceil(total_durs * hparams['audio_sample_rate'] / hparams['hop_size'])
            id_and_sizes.append((item_name, total_frames))
        get_size = lambda x: x[1]
        bs = batch_by_size(id_and_sizes, get_size, max_tokens=max_tokens, max_sentences=bsz)
        for i in range(len(bs)):
            bs[i] = [bs[i][j][0] for j in range(len(bs[i]))]

        for batch in tqdm(bs, total=len(bs),
                              desc='Extracting' + (f' {task_id}' if task_id >= 0 else '')):
            wavs, mel_lengths, lengths = [], [], []
            for item_name in batch:
                item = items[item_name]
                wav_fn = item['wav_fn']
                # (wav, _), mel = BaseBinarizer.process_audio(wav_fn, item, binarization_args)
                wav, _ = librosa.core.load(wav_fn, sr=hparams['audio_sample_rate'])
                # wav = pad_or_cut_xd(torch.Tensor(wav), math.ceil(wav.shape[0] / hparams['hop_size']) * hparams['hop_size']).numpy()
                wavs.append(wav)
                mel_lengths.append(math.ceil((wav.shape[0] + 1) / hparams['hop_size']))
                # lengths.append((wav.shape[0] + rmvpe.mel_extractor.hop_length - 1) // rmvpe.mel_extractor.hop_length)
                lengths.append((wav.shape[0] + hparams['hop_size'] - 1) // hparams['hop_size'])

            with torch.no_grad():
                f0s, uvs = rmvpe.get_pitch_batch(
                    wavs, sample_rate=hparams['audio_sample_rate'],
                    hop_size=hparams['hop_size'],
                    lengths=lengths,
                    fmax=hparams['f0_max'],
                    fmin=hparams['f0_min']
                )
            for idx in range(len(f0s)):
                f0_dict[batch[idx]] = f0s[idx]

        return f0_dict

    def process(self, a):
        self.load_meta_data()
        os.makedirs(self.save_dir, exist_ok=True)
        pe = hparams.get('pe', 'pw')

        if a.num_shards == 1:
            print('| Generate in 1 shard')
            if a.batch_size == 1:
                f0_dict = F0Extractor.generate(self.items, hparams, self.binarization_args, self.device)
            elif a.batch_size > 1:
                print(f'| Generate in max sentences {a.batch_size}, max tokens {a.max_tokens}')
                f0_dict = F0Extractor.generate_batch(self.items, hparams, self.binarization_args, self.device,
                                                     bsz=a.batch_size, max_tokens=a.max_tokens)
            save_path = os.path.join(self.save_dir, f"f0_{pe}_hop{hparams['hop_size']}_sr{hparams['audio_sample_rate']}.npy")
        elif a.num_shards > 1 and a.shard_id == -1:
            # NOTE: this doesn't work
            print(f'| Generate {a.num_shards} shards in parallel')
            f0_dict = {}
            pool = Pool(min(a.num_shards, int(os.getenv('N_PROC', os.cpu_count()))))
            shard_size = int(np.ceil(len(self.item_names) / a.num_shards))
            res = []
            for shard_id in range(a.num_shards):
                shard_items = {item_name: self.items[item_name] for item_name in self.item_names[shard_id * shard_size: (shard_id + 1) * shard_size]}
                r = pool.apply_async(F0Extractor.generate, args=[shard_items, hparams, self.binarization_args, self.device, shard_id])
                res.append(r)
            pool.close()
            pool.join()
            for r in res:
                s = r.get()
                f0_dict = {**f0_dict, **s}
            save_path = os.path.join(self.save_dir, f"f0_{pe}_hop{hparams['hop_size']}_sr{hparams['audio_sample_rate']}.npy")
        elif a.num_shards > 1 and a.shard_id >= 0:
            print(f'| Generate shard {a.shard_id}/{a.num_shards}')
            shard_size = int(np.ceil(len(self.item_names) / a.num_shards))
            shard_items = {item_name: self.items[item_name] for item_name in self.item_names[a.shard_id * shard_size: (a.shard_id + 1) * shard_size]}
            f0_dict = F0Extractor.generate(shard_items, hparams, self.binarization_args, self.device, a.shard_id)
            save_path = os.path.join(self.save_dir, f"f0_{pe}_hop{hparams['hop_size']}_sr{hparams['audio_sample_rate']}_shard{a.shard_id}.npy")

        # save_path = os.path.join(self.save_dir, f"f0_{pe}_hop{hparams['hop_size']}_sr{hparams['audio_sample_rate']}.npy")
        np.save(save_path, f0_dict, allow_pickle=True)

        return f0_dict

# %%
if __name__ == '__main__':
    # set_hparams('/mnt/sdb/liruiqi/SingingDictation/research/rme/config/base_me.yaml')
    # f0_extractor = F0Extractor('/mnt/sdb/liruiqi/SingingDictation/data/processed/m4singer')
    # set_hparams('/mnt/sdb/liruiqi/SingingDictation/research/rme/config/base_me2.yaml')
    # f0_extractor = F0Extractor('/mnt/sdb/liruiqi/SingingDictation/data/processed/m4+rms')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--batch-size', '-bsz', required=False, default=1, type=int)
    parser.add_argument('--max-tokens', required=False, default=100000, type=int)
    parser.add_argument('--num-shards', required=False, default=1, type=int)
    parser.add_argument('--shard-id', required=False, default=-1, type=int)
    a = parser.parse_args()

    set_hparams(a.config)
    f0_extractor = F0Extractor(a.save_dir)
    f0_extractor.process(a)

# %%
# research/rme/scripts/extract_f0.py \
# --config /mnt/sdb/liruiqi/SingingDictation/research/rme/config/base_me2.yaml \
# --save-dir /mnt/sdb/liruiqi/SingingDictation/data/processed/m4+rms/

# CUDA_VISIBLE_DEVICES=0 python research/rme/scripts/extract_f0.py \
# --config /mnt/sdb/liruiqi/SingingDictation/singing/svs/config/base_rms_xiaoma.yaml \
# --save-dir /mnt/sdb/liruiqi/datasets/xiaoma/ \
# --batch-size 40 \
# --max-tokens 160000

# rms
# CUDA_VISIBLE_DEVICES=0 python research/rme/scripts/extract_f0.py \
# --config /mnt/sdb/liruiqi/RMSSinger_dev/singing/svs/config/base_rms.yaml \
# --save-dir /mnt/sdb/liruiqi/RMSSinger_dev/data/processed/rms \
# --batch-size 40 \
# --max-tokens 160000

# m4
# CUDA_VISIBLE_DEVICES=0 python research/rme/scripts/extract_f0.py \
# --config /mnt/sdb/liruiqi/RMSSinger_dev/singing/svs/config/base_rms_m4.yaml \
# --save-dir /mnt/sdb/liruiqi/RMSSinger_dev/data/processed/m4 \
# --batch-size 40 \
# --max-tokens 160000

# speech2singing
# CUDA_VISIBLE_DEVICES=0 python research/rme/scripts/extract_f0.py \
# --config /mnt/sdb/liruiqi/RMSSinger_dev/singing/svs/config/base_rms_sts.yaml \
# --save-dir /mnt/sdb/liruiqi/RMSSinger_dev/data/processed/sts \
# --batch-size 40 \
# --max-tokens 160000

# 英文segment
# CUDA_VISIBLE_DEVICES=0 python research/rme/scripts/extract_f0.py \
# --config /mnt/sdb/liruiqi/RMSSinger_dev/singing/svs/config/base_rms_en.yaml \
# --save-dir /mnt/sdb/liruiqi/RMSSinger_dev/data/processed/en \
# --batch-size 40 \
# --max-tokens 160000

# %%
# rank=3
# CUDA_VISIBLE_DEVICES=$rank python research/rme/scripts/extract_f0.py \
# --config /mnt/sdb/liruiqi/SingingDictation/research/rme/config/base_me3.yaml \
# --save-dir /mnt/sdb/liruiqi/SingingDictation/data/processed/m4+rms2/ \
# --num-shards 4 \
# --shard-id $rank

# CUDA_VISIBLE_DEVICES=0 python research/rme/scripts/extract_f0.py \
# --config /mnt/sdb/liruiqi/SingingDictation/research/rme/config/base_me3.yaml \
# --save-dir /mnt/sdb/liruiqi/SingingDictation/data/processed/m4+rms2/ \
# --num-shards 4 \
# --shard-id -1

# 尝试 batch
# python research/rme/scripts/extract_f0.py \
# --config /mnt/sdb/liruiqi/SingingDictation/research/rme/config/base_me2.yaml \
# --save-dir /mnt/sdb/liruiqi/SingingDictation/data/processed/m4+rms-batch \
# --batch-size 32

# CUDA_VISIBLE_DEVICES=1 python research/rme/scripts/extract_f0.py \
# --config /mnt/sdb/liruiqi/SingingDictation/research/rme/config/base_me2.yaml \
# --save-dir /mnt/sdb/liruiqi/SingingDictation/data/processed/m4+rms-temp/

# %%
# items = {}
# item_names = []
# metafile_path = '/mnt/sdb/liruiqi/SingingDictation/data/processed/m4singer/metadata.json'
# items_list = json.load(open(metafile_path))
# for r in tqdm(items_list, desc='Loading meta data.'):
#     item_name = r['item_name']
#     items[item_name] = r
#     item_names.append(item_name)
#
# # %%
# hparams = set_hparams('/mnt/sdb/liruiqi/SingingDictation/research/rme/config/base_me.yaml')
# binarization_args = hparams['binarization_args']
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # %%
# item_name = item_names[0]
# item = items[item_name]
# wav_fn = item['wav_fn']
# wav, mel = BaseBinarizer.process_audio(wav_fn, item, binarization_args)
#
# f0 = None
# pe = hparams.get('pe', 'pw')
# wav, length = item['wav'], item['mel'].shape[0]
# if rmvpe is None:
#     rmvpe = RMVPE(hparams['pe_ckpt'], device=device)
# with torch.no_grad():
#     f0, uv = rmvpe.get_pitch(
#         wav, sample_rate=hparams['audio_sample_rate'],
#         hop_size=rmvpe.mel_extractor.hop_length,
#         length=(wav.shape[0] + rmvpe.mel_extractor.hop_length - 1) // rmvpe.mel_extractor.hop_length,
#         interp_uv=True
#     )
#     f0[uv] = 0
#     f0 = resample_align_curve(
#         f0,
#         original_timestep=rmvpe.mel_extractor.hop_length / hparams['audio_sample_rate'],
#         target_timestep=hparams['hop_size'] / hparams['audio_sample_rate'],
#         align_length=length
#     )
#
# # %%
# f0, _ = pw.harvest(wav.astype(np.double), hparams['audio_sample_rate'], frame_period=hparams['hop_size'] * 1000 / hparams['audio_sample_rate'])
# delta_l = length - len(f0)
# if delta_l < 0:
#     f0 = f0[:length]
# elif delta_l > 0:
#     f0 = np.concatenate((f0, np.full(delta_l, fill_value=f0[-1])), axis=0)
#
# # %%
# import matplotlib.pyplot as plt
# plt.plot(f0_rmvpe)
# plt.plot(f0_pw)
# plt.show()
