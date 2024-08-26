import random, os
import subprocess
from copy import deepcopy
import logging
import json
from functools import partial
from pathlib import Path
import glob
import numpy as np
from tqdm import tqdm
import pyworld as pw
import torch
from utils.audio import librosa_wav2spec

from data_gen.tts.base_binarizer import BaseBinarizer, BinarizationError
from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDatasetBuilder
from utils.commons.multiprocess_utils import multiprocess_run_tqdm, multiprocess_run_cuda, chunked_multiprocess_run
from utils.audio.align import mel2token_to_dur
from utils.text.text_encoder import build_token_encoder
from utils.audio.pitch_utils import f0_to_coarse, resample_align_curve, hz_to_midi
from utils.audio import get_energy_librosa, get_breathiness_pyworld, get_zcr_librosa
from utils.commons.dataset_utils import pad_or_cut_xd
from modules.pe.rmvpe import RMVPE
import modules.pe.rmvpe.extractor as f0_extractor

# rmvpe = None
# f0_dict, f0_hopsize, f0_sr = None, None, None

class TechExtractionBinarizer(BaseBinarizer):
    def __init__(self, processed_data_dir=None):
        super(TechExtractionBinarizer, self).__init__(processed_data_dir)
        # self.spk_map = json.load(open(os.path.join(hparams["processed_data_dir"], "spk_map.json")))
        self.spk_map = set()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def split_train_test_set(self, item_names):
        item_names = deepcopy(item_names)
        test_item_names = [x for x in item_names if any([ts in x for ts in hparams['test_prefixes']])]
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    def load_meta_data(self):
        metafile_paths = glob.glob(f"{self.processed_data_dir}/*/metadata.json")
        ds_names = "your_dataset_name"
        #metafile_path = hparams.get('metafile_path', f"{self.processed_data_dir}/metadata.json")
        #if ',' in metafile_path:
        #    metafile_paths = metafile_path.split(',')
        #    ds_names = hparams.get('ds_names', ','.join([str(i) for i in range(len(metafile_paths))])).split(',')
        #else:
        #    metafile_paths = [metafile_path]
        #    ds_names = [hparams.get('ds_names', '0')]
        for idx, metafile_path in enumerate(metafile_paths):
            items_list = json.load(open(metafile_path))
            for r in tqdm(items_list, desc=f'| Loading meta data for dataset {ds_names} {idx}.'):
                # print(r)
                item_name = r['item_name']
                if item_name in self.items:
                    print(f'warning: item name {item_name} duplicated')
                self.items[item_name] = r
                self.item_names.append(item_name)
                self.items[item_name]['ds_name'] = ds_names
                self.items[item_name]['wav_fn'] = os.path.join(f"{self.processed_data_dir}", self.items[item_name]['wav_fn'])
                self.spk_map.add(r['singer'])
        if self.binarization_args['shuffle']:
            random.seed(hparams.get('seed', 42))
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._test_item_names

    @property
    def test_item_names(self):
        return self._test_item_names

    def process(self):
        self.load_meta_data()
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)

        self.spk_map.add('_others')
        self.spk_map = sorted(list(self.spk_map))
        self.spk_map = {self.spk_map[idx]: idx for idx in range(len(self.spk_map))}
        json.dump(self.spk_map, open(f"{hparams['binary_data_dir']}/spk_map.json", 'w'), indent=2)

        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')
    @classmethod
    def process_audio(cls, wav_fn, res, binarization_args):
        wav2spec_dict = librosa_wav2spec(
            wav_fn,
            fft_size=hparams['fft_size'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'])
        mel = wav2spec_dict['mel']
        wav = wav2spec_dict['wav'].astype(np.float16)
        wav_orig = wav2spec_dict['wav_orig'].astype(np.float16)
        # wav = wav2spec_dict['wav']
        if binarization_args['with_linear']:
            res['linear'] = wav2spec_dict['linear']
        res.update({'mel': mel, 'wav': wav, 'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0]})
        return (wav, wav_orig), mel
    @torch.no_grad()
    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']
        args = []
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        process_item = partial(self.process_item, binarization_args=self.binarization_args)
        lengths = []
        total_sec = 0
        meta_data = list(self.meta_data(prefix))
        args = [item for item in meta_data]

        # extract f0
        if hparams.get('pe', 'pw') == 'rmvpe':
            wav_fns = [item['wav_fn'] for item in args]
            f0s = f0_extractor.extract(wav_fns, ckpt=hparams.get('pe_ckpt', None), sr=hparams['audio_sample_rate'],
                                       hop_size=hparams['hop_size'], fmax=hparams['f0_max'], fmin=hparams['f0_min'],
                                       device='cuda')
            args = [{**item, **{'f0': f0s[idx]}} for idx, item in enumerate(args)]

        for item_id, (_, item) in enumerate(
                zip(tqdm(meta_data, desc='| Processing data', total=len(args)),
                    chunked_multiprocess_run(process_item, args))):
            if item is None:
                continue
            if not self.binarization_args['with_wav'] and 'wav' in item:
                del item['wav']
            if not self.binarization_args['with_mel'] and 'mel' in item:
                del item['mel']

            builder.add_item(item)
            lengths.append(item['len'])
            total_sec += item['sec']
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    @torch.no_grad()
    def process_item(self, item, binarization_args):
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        (wav, wav_orig), mel = self.process_audio(wav_fn, item, binarization_args)
        item['wav'] = wav = wav_orig    # need to be wav_orig, since we make mel again
        length = mel.shape[0]
        try:
            item["spk_id"] = self.spk_map[item["singer"]]

            if hparams.get('f0_filepath', '') == '':
                f0 = None
                pe = hparams.get('pe', 'pw')
                if pe == 'rmvpe':
                    pass
                elif pe == 'pw':
                    f0, _ = pw.harvest(wav.astype(np.double), hparams['audio_sample_rate'],
                                       frame_period=hparams['hop_size'] * 1000 / hparams['audio_sample_rate'])
                    delta_l = length - len(f0)
                    if delta_l < 0:
                        f0 = f0[:length]
                    elif delta_l > 0:
                        f0 = np.concatenate((f0, np.full(delta_l, fill_value=f0[-1])), axis=0)
            else:
                global f0_dict, f0_hopsize, f0_sr
                # NOTE: f0 的 sr 和 hopsize，必须和本 config 所设定的一致！
                if f0_dict is None:
                    f0_filepaths = hparams.get('f0_filepath', '')
                    if ',' in f0_filepaths:
                        f0_filepaths = f0_filepaths.split(',')
                        f0_dict = {}
                        for f0_filepath in f0_filepaths:
                            _f0_dict = np.load(f0_filepath, allow_pickle=True).item()
                            f0_dict = {**f0_dict, **_f0_dict}
                    else:
                        f0_dict = np.load(f0_filepaths, allow_pickle=True).item()
                f0 = f0_dict[item_name]
                if abs(f0.shape[0] - length) < 5:
                    f0 = pad_or_cut_xd(torch.from_numpy(f0), length).numpy()
                assert len(f0) == length

            if f0 is not None:
                item['f0'] = f0

            mel2ph, mel2word, dur_word = self.process_align(item["ph_durs"], mel, item.get('ph2words', None))
            item['mel2ph']= mel2ph
            if mel2word!= None and mel2word[0] == 0:    # better start from 1, consistent with mel2ph
                mel2word = [i + 1 for i in mel2word]
                item['mel2word'], item['dur_word'] = mel2word, dur_word

            if binarization_args.get('with_breathiness', False):
                breathiness = get_breathiness_pyworld(wav, item['f0'], length, hparams)
                item['breathiness'] = breathiness

            if binarization_args.get('with_energy', False):
                energy = get_energy_librosa(wav, length, hparams)
                item['energy'] = energy

            if binarization_args.get('with_zcr', False):
                zcr = get_zcr_librosa(wav, length, hparams)
                item['zcr'] = zcr

        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return item

    @staticmethod
    def process_align(ph_durs, mel, ph2words= None, hop_size=hparams['hop_size'], audio_sample_rate=hparams['audio_sample_rate']):
        mel2ph = np.zeros([mel.shape[0]], int)
        startTime = 0

        for i_ph in range(len(ph_durs)):
            start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
            end_frame = int((startTime + ph_durs[i_ph]) * audio_sample_rate / hop_size + 0.5)
            mel2ph[start_frame:end_frame] = i_ph + 1
            startTime = startTime + ph_durs[i_ph]

        mel2word, dur_word = None, None
        if ph2words!= None:
            mel2word = [ph2words[p - 1] for p in mel2ph]
            dur_word = mel2token_to_dur(mel2word, max(ph2words))
            dur_word = dur_word.tolist()
        return mel2ph, mel2word, dur_word

class OutDomainTechExtractionBinarizer(TechExtractionBinarizer):
    def load_meta_data(self):
        metafile_paths = glob.glob(f"{self.processed_data_dir}/meta.json")
        ds_names = "your_dataset_name"
        #metafile_path = hparams.get('metafile_path', f"{self.processed_data_dir}/metadata.json")
        #if ',' in metafile_path:
        #    metafile_paths = metafile_path.split(',')
        #    ds_names = hparams.get('ds_names', ','.join([str(i) for i in range(len(metafile_paths))])).split(',')
        #else:
        #    metafile_paths = [metafile_path]
        #    ds_names = [hparams.get('ds_names', '0')]
        for idx, metafile_path in enumerate(metafile_paths):
            items_list = json.load(open(metafile_path))
            for r in tqdm(items_list, desc=f'| Loading meta data for dataset {ds_names} {idx}.'):
                item_name = r['item_name']
                if item_name in self.items:
                    print(f'warning: item name {item_name} duplicated')
                self.items[item_name] = r
                self.item_names.append(item_name)
                self.items[item_name]['ds_name'] = ds_names
                self.items[item_name]['wav_fn'] = os.path.join(f"{self.processed_data_dir}", self.items[item_name]['wav_fn'])
                self.spk_map.add(r['singer'])
        if self.binarization_args['shuffle']:
            random.seed(hparams.get('seed', 42))
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names = [], deepcopy(self.item_names)

    def split_train_test_set(self, item_names):
        return [], item_names

    def process(self):
        self.load_meta_data()
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)

        self.spk_map.add('_others')
        self.spk_map = sorted(list(self.spk_map))
        self.spk_map = {self.spk_map[idx]: idx for idx in range(len(self.spk_map))}
        json.dump(self.spk_map, open(f"{hparams['binary_data_dir']}/spk_map.json", 'w'), indent=2)

        self.process_data('test')
