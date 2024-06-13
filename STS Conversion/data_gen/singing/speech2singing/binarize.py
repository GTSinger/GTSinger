import os
import subprocess
import sys

from data_gen.tts.base_binarizer import BaseBinarizer, BinarizationError
import re
from copy import deepcopy
import logging
from data_gen.tts.binarizer_zh import ZhBinarizer
from utils.hparams import hparams, set_hparams

from utils.multiprocess_utils import chunked_multiprocess_run
import random
import traceback
import json
from resemblyzer import VoiceEncoder
from tqdm import tqdm
from data_gen.tts.data_gen_utils import get_mel2ph, get_pitch, build_phone_encoder
import numpy as np
from utils.indexed_datasets import IndexedDatasetBuilder
from vocoders.base_vocoder import get_vocoder_cls
import pandas as pd
import parselmouth

def split_train_test_set(item_names):
    item_names = deepcopy(item_names)
    test_item_names = [x for x in item_names if any([ts in x for ts in hparams['test_prefixes']])]
    train_item_names = [x for x in item_names if x not in set(test_item_names)]
    logging.info("train {}".format(len(train_item_names)))
    logging.info("test {}".format(len(test_item_names)))
    return train_item_names, test_item_names

class Speech2SingingBinarizer(BaseBinarizer):
    def __init__(self, processed_data_dir=None):
        # super(Speech2SingingBinarizer, self).__init__()
        if processed_data_dir is None:
            processed_data_dir = hparams['processed_data_dir']
        self.processed_data_dirs = processed_data_dir.split(",")
        self.binarization_args = hparams['binarization_args']
        self.pre_align_args = hparams['pre_align_args']
        self.forced_align = self.pre_align_args['forced_align']
        tg_dir = None
        if self.forced_align == 'mfa':
            tg_dir = 'mfa_outputs'
        if self.forced_align == 'kaldi':
            tg_dir = 'kaldi_outputs'
        self.item2txt = {}
        self.item2ph = {}
        self.item2sing_wavfn = {}
        self.item2speech_wavfn = {}
        self.item2sing_tgfn = {}
        self.item2speech_tgfn = {}
        self.item2spk = {}
        # below are speech data
        self.item2speech_spenv_fn = {}
        self.item2speech_spmel_mono_fn = {}
        self.item2speech_wav_mono_fn = {}
        self.item2sing_spenv_fn = {}
        self.item2speech_wav2vec2_mono_fn = {}
        self.item2sing_wav2vec2_fn = {}
        for ds_id, processed_data_dir in enumerate(self.processed_data_dirs):
            self.meta_df = pd.read_csv(f"{processed_data_dir}/metadata_phone.csv", dtype=str)
            for r_idx, r in tqdm(self.meta_df.iterrows(), desc='Loading meta data.'):
                item_name = raw_item_name = r['item_name']
                if len(self.processed_data_dirs) > 1:
                    item_name = f'ds{ds_id}_{item_name}'
                if r.get('spk', 'SPK1') not in set(hparams['spk_list']):    # skip some spk with so few data
                    continue
                self.item2txt[item_name] = r['txt']
                self.item2ph[item_name] = r['ph']
                self.item2sing_wavfn[item_name] = r['sing_wav_fn']
                self.item2speech_wavfn[item_name] = r['speech_wav_fn']
                self.item2spk[item_name] = r.get('spk', 'SPK1')
                if len(self.processed_data_dirs) > 1:
                    self.item2spk[item_name] = f"ds{ds_id}_{self.item2spk[item_name]}"
                # if tg_dir is not None:
                #     self.item2tgfn[item_name] = f"{processed_data_dir}/{tg_dir}/{raw_item_name}.TextGrid"
                spk, song_name, song_idx = item_name.split("#")
                self.item2sing_tgfn[item_name] = f"{processed_data_dir}/textgrid/{'#'.join([spk, song_name, '专业'])}" \
                                                 f"/{'#'.join([spk, song_name, '专业', song_idx])}.TextGrid"
                self.item2speech_tgfn[item_name] = f"{processed_data_dir}/textgrid/{'#'.join([spk, song_name, '歌词'])}" \
                                                   f"/{'#'.join([spk, song_name, '歌词', song_idx])}.TextGrid"
                self.item2speech_spenv_fn[item_name] = f"{processed_data_dir}/spenv/{'#'.join([spk, song_name, '歌词'])}" \
                                                f"/{'#'.join([spk, song_name, '歌词', song_idx])}.npy"
                self.item2speech_spmel_mono_fn[item_name] = f"{processed_data_dir}/spmel_mono/{'#'.join([spk, song_name, '歌词'])}" \
                                                     f"/{'#'.join([spk, song_name, '歌词', song_idx])}.npy"
                if self.binarization_args["with_wav_mono"]:
                    self.item2speech_wav_mono_fn[item_name] = f"{processed_data_dir}/wav_mono/{'#'.join([spk, song_name, '歌词'])}" \
                                     f"/{'#'.join([spk, song_name, '歌词', song_idx])}.npy"
                self.item2sing_spenv_fn[item_name] = f"{processed_data_dir}/spenv/{'#'.join([spk, song_name, '专业'])}" \
                                     f"/{'#'.join([spk, song_name, '专业', song_idx])}.npy"
                self.item2speech_wav2vec2_mono_fn[item_name] = f"{processed_data_dir}/wav2vec2_mono/{'#'.join([spk, song_name, '歌词'])}" \
                                                     f"/{'#'.join([spk, song_name, '歌词', song_idx])}.npy"
                self.item2sing_wav2vec2_fn[item_name] = f"{processed_data_dir}/wav2vec2_mono/{'#'.join([spk, song_name, '专业'])}" \
                                                     f"/{'#'.join([spk, song_name, '专业', song_idx])}.npy"
        self.item_names = sorted(list(self.item2txt.keys()))
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names = split_train_test_set(self.item_names)

        if self.binarization_args.get('with_bpe', False):
            self.make_bpe()

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._test_item_names

    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            ph = self.item2ph[item_name]
            txt = self.item2txt[item_name]
            speech_tg_fn = self.item2speech_tgfn.get(item_name)
            sing_tg_fn = self.item2sing_tgfn.get(item_name)
            speech_wav_fn = self.item2speech_wavfn[item_name]
            sing_wav_fn = self.item2sing_wavfn[item_name]
            spk_id = self.item_name2spk_id(item_name)
            speech_spenv_fn = self.item2speech_spenv_fn[item_name]
            speech_spmel_mono_fn = self.item2speech_spmel_mono_fn[item_name]
            speech_wav_mono_fn = self.item2speech_wav_mono_fn[item_name] if self.binarization_args["with_wav_mono"] else None
            sing_spenv_fn = self.item2sing_spenv_fn[item_name]
            speech_wav2vec2_mono_fn = self.item2speech_wav2vec2_mono_fn[item_name]
            sing_wav2vec2_fn = self.item2sing_wav2vec2_fn[item_name]
            if not hparams['binarization_args'].get('with_sing_spenv', False):
                sing_spenv_fn = None
            if not self.binarization_args.get('with_speech_wav2vec2_mono', False):
                speech_wav2vec2_mono_fn = None
            yield item_name, ph, txt, (speech_tg_fn, sing_tg_fn), (speech_wav_fn, sing_wav_fn), spk_id, \
                (speech_spenv_fn, speech_spmel_mono_fn, speech_wav_mono_fn, sing_spenv_fn, speech_wav2vec2_mono_fn, sing_wav2vec2_fn)

    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']
        args = []
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        speech_lengths = []
        sing_lengths = []
        speech_f0s = []
        sing_f0s = []
        total_sec = 0
        if self.binarization_args['with_spk_embed']:
            try:
                voice_encoder = VoiceEncoder().cuda()
            except:
                voice_encoder = VoiceEncoder()

        meta_data = list(self.meta_data(prefix))
        for m in meta_data:
            args.append(list(m) + [self.phone_encoder, self.binarization_args])
        num_workers = self.num_workers
        for f_id, (_, item) in enumerate(
                zip(tqdm(meta_data), chunked_multiprocess_run(self.process_item, args, num_workers=num_workers))):
        # for f_id, arg in enumerate(args):     # NOTE: this for debug
        #     item = self.process_item(*arg)
            if item is None:
                continue
            item['spk_embed'] = voice_encoder.embed_utterance(item['speech']['wav']) \
                if self.binarization_args['with_spk_embed'] else None
            if not self.binarization_args['with_wav'] and 'wav' in item['speech']:
                del item['speech']['wav']
            if not self.binarization_args['with_wav'] and 'wav' in item['sing']:
                del item['sing']['wav']
            if not self.binarization_args['with_wav_mono'] and 'wav_mono' in item['speech']:
                del item['speech']['wav_mono']
            if not self.binarization_args['with_wav_mono'] and 'wav_mono' in item['sing']:
                del item['sing']['wav_mono']
            if not self.binarization_args.get('with_sing_spenv', False) and 'spenv' in item['sing']:
                del item['sing']['spenv']
            if not self.binarization_args.get('with_speech_wav2vec2_mono', False) and 'wav2vec2_mono' in item['speech']:
                del item['speech']['wav2vec2_mono']
            if not self.binarization_args.get('with_sing_wav2vec2', False) and 'wav2vec2' in item['sing']:
                del item['sing']['wav2vec2']
            if not self.binarization_args.get('with_speech_mel', True) and 'mel' in item['speech']:
                del item['speech']['mel']
            if not self.binarization_args.get('with_speech_spenv', False) and 'spenv' in item['speech']:
                del item['speech']['spenv']
            builder.add_item(item)
            speech_lengths.append(item['speech']['len'])     # NOTE: 注意，这里的length只加了singing的长度，which一般比speech长
            sing_lengths.append(item['sing']['len'])     # NOTE: 注意，这里的length只加了singing的长度，which一般比speech长
            total_sec += item['speech']['sec'] + item['sing']['sec']
            if item['speech'].get('f0') is not None:
                speech_f0s.append(item['speech']['f0'])
            if item['sing'].get('f0') is not None:
                sing_f0s.append(item['sing']['f0'])
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_speech_lengths.npy', speech_lengths)
        np.save(f'{data_dir}/{prefix}_sing_lengths.npy', sing_lengths)
        if len(speech_f0s) > 0:
            f0s = np.concatenate(speech_f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(f'{data_dir}/{prefix}_speech_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
        if len(sing_f0s) > 0:
            f0s = np.concatenate(sing_f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(f'{data_dir}/{prefix}_sing_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, feats, encoder, binarization_args):
        speech_tg_fn, sing_tg_fn = tg_fn
        speech_wav_fn, sing_wav_fn = wav_fn
        speech_spenv_fn, speech_spmel_mono_fn, speech_wav_mono_fn, sing_spenv_fn, speech_wav2vec2_mono_fn, sing_wav2vec2_fn = feats
        res = {
            'item_name': item_name,
            'txt': txt,
            'ph': ph,
            'speech': {
                'wav_fn': speech_wav_fn,
                'tg_fn': speech_tg_fn
            },
            'sing': {
                'wav_fn': sing_wav_fn,
                'tg_fn': sing_tg_fn
            },
            'spk_id': spk_id
        }
        try:
            if binarization_args['with_linear']:
                speech_wav, speech_mel, speech_linear_stft = get_vocoder_cls(hparams).wav2spec(speech_wav_fn,
                                                                                               return_linear=True)
                res['speech']['linear'] = speech_linear_stft
                sing_wav, sing_mel, sing_linear_stft = get_vocoder_cls(hparams).wav2spec(sing_wav_fn,
                                                                                         return_linear=True)
                res['sing']['linear'] = sing_linear_stft
            else:
                speech_wav, speech_mel = get_vocoder_cls(hparams).wav2spec(speech_wav_fn)
                sing_wav, sing_mel = get_vocoder_cls(hparams).wav2spec(sing_wav_fn)
            speech_spenv = np.load(speech_spenv_fn)
            speech_spmel_mono = np.load(speech_spmel_mono_fn)
            speech_wav_mono = np.load(speech_wav_mono_fn) if hparams['binarization_args']["with_wav_mono"] else None
            sing_spenv = np.load(sing_spenv_fn) if hparams['binarization_args'].get('with_sing_spenv', False) else None
            speech_wav2vec2_mono = np.load(speech_wav2vec2_mono_fn) if hparams['binarization_args'].get('with_speech_wav2vec2_mono', False) else None
            sing_wav2vec2 = np.load(sing_wav2vec2_fn) if hparams['binarization_args'].get('with_sing_wav2vec2', False) else None
            res['speech'].update({
                'mel': speech_mel,
                'wav': speech_wav,
                'spenv': speech_spenv,
                'spmel_mono': speech_spmel_mono,
                'wav_mono': speech_wav_mono,
                'wav2vec2_mono': speech_wav2vec2_mono,
                'sec': len(speech_wav) / hparams['audio_sample_rate'],
                'len': speech_mel.shape[0],
            })
            res['sing'].update({
                'mel': sing_mel,
                'wav': sing_wav,
                'spenv': sing_spenv,
                'wav2vec2': sing_wav2vec2,
                'sec': len(sing_wav) / hparams['audio_sample_rate'],
                'len': sing_mel.shape[0]
            })
            if binarization_args['with_f0']:
                cls.get_pitch(speech_wav, speech_mel, res['speech'])
                if binarization_args['with_f0cwt']:
                    cls.get_f0cwt(res['speech']['f0'], res['speech'])
                cls.get_pitch(sing_wav, sing_mel, res['sing'])
                if binarization_args['with_f0cwt']:
                    cls.get_f0cwt(res['sing']['f0'], res['sing'])
            if binarization_args['with_txt'] and binarization_args.get('with_ph', True):
                try:
                    phone_encoded = res['phone'] = encoder.encode(ph)
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    cls.get_align(speech_tg_fn, ph, speech_mel, phone_encoded, res['speech'])
                    cls.get_align(sing_tg_fn, ph, sing_mel, phone_encoded, res['sing'])
                    if binarization_args['trim_eos_bos']:
                        bos_dur = res['speech']['dur'][0]
                        eos_dur = res['speech']['dur'][-1]
                        res['speech']['mel'] = speech_mel[bos_dur:-eos_dur]
                        res['speech']['f0'] = res['speech']['f0'][bos_dur:-eos_dur]
                        res['speech']['pitch'] = res['speech']['pitch'][bos_dur:-eos_dur]
                        res['speech']['mel2ph'] = res['speech']['mel2ph'][bos_dur:-eos_dur]
                        res['speech']['wav'] = speech_wav[bos_dur * hparams['hop_size']:-eos_dur * hparams['hop_size']]
                        res['speech']['dur'] = res['speech']['dur'][1:-1]
                        bos_dur = res['sing']['dur'][0]
                        eos_dur = res['sing']['dur'][-1]
                        res['sing']['mel'] = sing_mel[bos_dur:-eos_dur]
                        res['sing']['f0'] = res['sing']['f0'][bos_dur:-eos_dur]
                        res['sing']['pitch'] = res['sing']['pitch'][bos_dur:-eos_dur]
                        res['sing']['mel2ph'] = res['sing']['mel2ph'][bos_dur:-eos_dur]
                        res['sing']['wav'] = sing_wav[bos_dur * hparams['hop_size']:-eos_dur * hparams['hop_size']]
                        res['sing']['dur'] = res['sing']['dur'][1:-1]
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        except parselmouth.PraatError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        except ValueError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        except FileNotFoundError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return res

    def make_bpe(self):
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        data_dir = hparams['binary_data_dir']
        for k, v in self.item2txt.items():
            v = ''.join([c for c in v if c.isalpha() or c == ' '])
            v = re.sub(r'\s+', ' ', v)
            self.item2txt[k] = v
        for split in ['train', 'valid', 'test']:
            with open(f"{data_dir}/{split}_sents.txt", 'w') as f:
                for n in getattr(self, f'{split}_item_names'):
                    f.write(f'{self.item2txt[n]}\n')
        if hparams.get('bpe_dir', '') != '':
            subprocess.check_call(
                f'cp {hparams["bpe_dir"]}/bpe.codes {data_dir}/bpe.codes', shell=True)
        else:
            subprocess.check_call(
                f'subword-nmt learn-bpe -s 1000 < {data_dir}/train_sents.txt > {data_dir}/bpe.codes', shell=True)
            print("| Learned bpe.")
        for split in ['train', 'valid', 'test']:
            subprocess.check_call(f'subword-nmt apply-bpe -c {data_dir}/bpe.codes '
                                  f'< {data_dir}/{split}_sents.txt '
                                  f'> {data_dir}/{split}_sents.txt.bpe', shell=True)
        print("| Applied bpe.")
        if hparams.get('bpe_dir', '') != '':
            subprocess.check_call(
                f'cp {hparams["bpe_dir"]}/bpe_set.json {data_dir}/bpe_set.json', shell=True)
        else:
            out = subprocess.check_output(f"subword-nmt get-vocab < {data_dir}/train_sents.txt.bpe", shell=True)
            out = out.decode("utf-8")
            voc = [v.split(" ")[0] for v in out.split("\n")]
            voc = ['<pad>', '<EOS>', '<UNK>'] + voc
            bpe_set_fn = f"{hparams['binary_data_dir']}/bpe_set.json"
            json.dump(voc, open(bpe_set_fn, 'w'))
            print("| Saved dict.")

        # for split in ['train', 'valid', 'test']:
        #     with open(f"{data_dir}/{split}_sents.txt.bpe", 'r') as f:
        #         item_names = getattr(self, f'{split}_item_names')
        #         lines = f.readlines()
        #         assert len(item_names) == len(lines), (split, len(item_names), len(lines))
        #         for n, l in zip(tqdm(item_names, desc='Updating item2txt'), lines):
        #             self.item2txt[n] = self.item2txt[n], l.strip()


class NHSS_Sp2Si_Binarizer(Speech2SingingBinarizer):
    pass

