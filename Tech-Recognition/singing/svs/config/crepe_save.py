# binarizer for rms singing

import random, os
from copy import deepcopy
import logging
from data_gen.tts.base_binarizer import BaseBinarizer, BinarizationError
from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDatasetBuilder
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from functools import partial
import numpy as np
from tqdm import tqdm
from utils.audio.align import mel2token_to_dur
import pyworld as pw
import json


class SingingBinarizer(BaseBinarizer):
    def __init__(self, processed_data_dir=None):
        super().__init__()


    def split_train_test_set(self, item_names):
        item_names = deepcopy(item_names)
        test_item_names = [x for x in item_names if any([ts in x for ts in hparams['test_prefixes']])]
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    def load_meta_data(self):
        processed_data_dir = self.processed_data_dir
        items_list = json.load(open(f"{processed_data_dir}/metadata.json"))
        for r in tqdm(items_list, desc='Loading meta data.'):
            item_name = r['item_name']
            self.items[item_name] = r
            self.item_names.append(item_name)
        if self.binarization_args['shuffle']:
            random.seed(1234)
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
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            yield self.items[item_name]

    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']
        args = []
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        process_item = partial(self.process_item, binarization_args=self.binarization_args)
        lengths = []
        total_sec = 0
        meta_data = list(self.meta_data(prefix))
        args = [{'item': item} for item in meta_data]
        for item_id, item in multiprocess_run_tqdm(process_item, args, desc='Processing data'):
            if item is None:
                continue
            if not self.binarization_args['with_wav'] and 'wav' in item:
                del item['wav']
            builder.add_item(item)
            lengths.append(item['len'])
            total_sec += item['sec']
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    @classmethod
    def process_item(cls, item, binarization_args):
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        wav, mel = cls.process_audio(wav_fn, item, binarization_args)
        item['spk_embed'] = np.load(wav_fn.replace(".wav", "_spk.npy"))
        try:
            cls.process_pitch(item, 0, 0)
            cls.process_align(item["tg_fn"], item)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return item
    
ii=0

# 拥有midi标注的数据，以及给定ph_durs 而不是 textgrid的数据
class CrepeSingingBinarizer(SingingBinarizer):
    # print(hparams)
    # ph_encoder = build_token_encoder(os.path.join(hparams["processed_data_dir"], "phone_set.json"))
    # spker_map = json.load(open(os.path.join(hparams["processed_data_dir"], "spker_set.json")))

    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']
        args = []
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        process_item = partial(self.process_item, binarization_args=self.binarization_args)
        lengths = []
        total_sec = 0
        meta_data = list(self.meta_data(prefix))
        args = [{'item': item} for item in meta_data]
        # print()
        # ii=0
        # self.ii=0
        for item_id, item in multiprocess_run_tqdm(process_item, args, desc='Processing data'):
            if item is None:
                # continue
                import sys
                sys.exit()
            # if not self.binarization_args['with_wav'] and 'wav' in item:
            #     del item['wav']
            builder.add_item(item)
            lengths.append(item['len'])
            total_sec += item['sec']
        builder.finalize()
        # np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    @classmethod
    def process_item(cls, item, binarization_args):
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        wav, mel = cls.process_audio(wav_fn, item, binarization_args)
        # item['spk_embed'] = np.load(wav_fn.replace(".wav", "_spk.npy"))
        # try:
            # item["ph_token"] = cls.ph_encoder.encode(' '.join(item["ph"]))
            # item["spk_id"] = cls.spker_map[item["singer"]]
            # cls.process_pitch(item, 0, 0)
            # if(os.path.exists(wav_fn.replace(".wav", ".npy"))):
                # item["f0"] = f0 = np.load(wav_fn.replace(".wav", ".npy"))[:mel.shape[0]]
            # else:
        _f0, t = pw.dio(wav.astype(np.double), hparams['audio_sample_rate'],
            frame_period=hparams['hop_size'] / hparams['audio_sample_rate'] * 1000.0,allowed_range=0.2)
        f0 = pw.stonemask(wav.astype(np.double), _f0, t, hparams['audio_sample_rate'])  # pitch refinement
        # v=f0>0
        # t,f0,c,_= crepe.predict(wav.astype(np.double),hparams['audio_sample_rate'],viterbi=True,
        #             step_size=hparams['hop_size'] / hparams['audio_sample_rate'] * 1000.0,verbose=0)
        # # item["f0"] = get_pitch_crepe(wav,mel,hparams)
        # # item["f0"] = f0 = np.load(wav_fn.replace(".wav", ".npy"))
        # f0[f0>1000]=1000
        # v= v[:len(f0)]
        # item["f0"] = f0 = f0 * v

        time_step = hparams['hop_size'] / hparams['audio_sample_rate'] * 1000
        f0_min = 80
        f0_max = 800

        if hparams['hop_size'] == 128:
            pad_size = 4
        elif hparams['hop_size'] == 256:
            pad_size = 2
        else:
            assert False

        # import parselmouth
        # f0 = parselmouth.Sound(wav, hparams['audio_sample_rate']).to_pitch_ac(
        #     time_step=time_step / 1000, voicing_threshold=0.1,
        #     pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
        # lpad = pad_size * 2
        # rpad = len(mel) - len(f0) - lpad
        # f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
        # delta_l = len(mel) - len(f0)
        # assert np.abs(delta_l) <= 8
        # if delta_l > 0:
        #     f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
        item["f0"] = f0 = f0[:len(mel)]

            # pit = harmof0.PitchTracker(sample_rate = hparams['audio_sample_rate']/2,hop_length = hparams['hop_size']/2, post_processing = True)
            # t, _f0,_,_ = pit.pred(wav.astype(np.double), hparams['audio_sample_rate'])
            # uv=_f0>0

            # f0 = f0[:len(_f0)]
            # item["f0"] = f0 = f0 * uv

        np.save(wav_fn.replace(".wav", "_midi.npy"),f0)
        # ii+=1
        # import os
        # if os.path.exists(wav_fn.replace(".wav", ".npy")):
        # #     global ii
        # #     ii+=1
        #     print(wav_fn.replace(".wav", ".npy"))
            # import sys
            # sys.exit()
        # print(ii,wav_fn.replace(".wav", ".npy"))
            # print(wav_fn.replace(".wav", ".npy"))
            # pitch_coarse = f0_to_coarse(f0)
            # item['pitch'] = pitch_coarse
            # cls.process_align(item["ph_durs"], mel, item)
        # except BinarizationError as e:
        #     print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
        #     return None
        return item
    
    @staticmethod
    def process_align(ph_durs, mel, item, hop_size=hparams['hop_size'], audio_sample_rate=hparams['audio_sample_rate']):
        mel2ph = np.zeros([mel.shape[0]], int)
        startTime = 0

        for i_ph in range(len(ph_durs)):
            start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
            end_frame = int((startTime + ph_durs[i_ph]) * audio_sample_rate / hop_size + 0.5)
            mel2ph[start_frame:end_frame] = i_ph + 1
            startTime = startTime + ph_durs[i_ph]

        item['mel2ph'] = mel2ph

        ph2word = item['ph2words']
        mel2word = [ph2word[p - 1] for p in item['mel2ph']]
        item['mel2word'] = mel2word  # [T_mel]
        dur_word = mel2token_to_dur(mel2word, max(item['ph2words']))
        item['dur_word'] = dur_word.tolist()  # [T_word]