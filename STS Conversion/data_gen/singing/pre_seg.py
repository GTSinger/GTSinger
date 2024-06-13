import os

os.environ["OMP_NUM_THREADS"] = "1"

from utils.multiprocess_utils import chunked_multiprocess_run
from vocoders.base_vocoder import VOCODERS
import traceback
from data_gen.tts.base_binarizer import BaseBinarizer
from data_gen.tts.base_pre_align import BasePreAlign
import pandas as pd
from tqdm import tqdm
import glob
import os
import re
import subprocess
from itertools import chain
import chardet
from data_gen.tts.data_gen_utils import trim_long_silences
from utils.audio import save_wav, to_mp3
from utils.hparams import hparams
from utils.text_norm import NSWNormalizer


def get_encoding(file):
    with open(file, 'rb') as f:
        return chardet.detect(f.read())['encoding']


class SingingPreSegPreAlign(BasePreAlign):
    def __init__(self):
        super().__init__()
        self.pre_align_args['hparams'] = hparams

    def meta_data(self):
        raw_data_dir = hparams['raw_data_dir']
        for singer in hparams['datasets']:
            if singer in hparams['short_datasets']:
                meta_fn = f'data/processed/{singer}/metadata_phone.csv'
                if not os.path.exists(meta_fn):
                    print(f"| Short dataset {singer} not found.")
                df = pd.read_csv(meta_fn)
                print(f"| {singer} (short): {len(df)}")
                for r_idx, r in df.iterrows():
                    item_name = r['item_name']
                    txt = "[TXT]" + r['txt']
                    wav_fn = r['wav_fn']
                    dir_name = 'speech'
                    song_name = item_name
                    yield f'{singer}#{dir_name}#{song_name}', wav_fn, (self.load_txt, txt), singer
            else:
                wav_fns = sorted(chain.from_iterable([
                    glob.glob(f'{raw_data_dir}/data/{singer}/*/*.{e}') for e in ['wav', 'mp3']]))
                print(f"| {singer}: {len(wav_fns)}")
                assert len(wav_fns) > 0
                for wav_fn in wav_fns:
                    song_name = os.path.basename(wav_fn)[:-4].replace("#", "_").replace(" ", "_")
                    dir_name = wav_fn.split("/")[-2]
                    txt_fn = wav_fn[:-4] + '.txt'
                    yield f'{singer}#{dir_name}#{song_name}', wav_fn, (self.load_txt, txt_fn), singer

    @staticmethod
    def load_txt(txt_fn):
        if '[TXT]' in txt_fn:
            text = [txt_fn[5:]]
        else:
            if not os.path.exists(txt_fn):
                return None
            encoding = get_encoding(txt_fn)
            if encoding == 'GB2312':
                encoding = 'GB18030'
            try:
                text = open(txt_fn, 'r', encoding=encoding).readlines()
            except:
                print(f"| file encoding error: {txt_fn}")
                return None
        text = [NSWNormalizer(x.strip()).normalize() for x in text]
        text = [re.sub('[^\u4e00-\u9fff]', '', x) for x in text]
        text = [re.sub('\s+', ' ', x).strip() for x in text]
        text = [x for x in text if x != '']
        return " SEP ".join(text)

    @staticmethod
    def process_wav(item_name, wav_fn, processed_dir, pre_align_args):
        out_path = f"{processed_dir}/wav_inputs/{item_name}"
        os.makedirs(f"{processed_dir}/wav_inputs", exist_ok=True)
        singer, dir_name, song_name = item_name.split("#")
        hparams = pre_align_args['hparams']
        if os.path.exists(f'{out_path}.mp3'):
            return f'{out_path}.mp3'

        if singer in hparams['short_datasets']:
            assert os.path.exists(wav_fn)
            subprocess.check_call(f'cp "{wav_fn}" "{out_path}.wav"', shell=True)
        else:
            try:
                wav, mask, sr = trim_long_silences(
                    wav_fn, sr=None, return_raw_wav=True,
                    vad_max_silence_length=pre_align_args['max_silence_length'])
            except:
                traceback.print_exc()
                print('| Empty wav: ', wav_fn)
                return None
            save_wav(wav[mask], out_path + '.wav', sr, False)
        to_mp3(out_path)
        return f'{out_path}.mp3'


class SingingPreSegBinarizer(BaseBinarizer):
    def __init__(self):
        super(SingingPreSegBinarizer, self).__init__()
        self.item_names = [x for x in self.item_names if x.split("#")[0] in hparams['datasets']]
        self.short_ds_to_seg_dir()

    def short_ds_to_seg_dir(self):
        os.makedirs(hparams['seg_out_dir'], exist_ok=True)
        args = []
        for r_idx, r in list(self.meta_df.iterrows()):
            item_name = r['item_name']
            singer, dir_name, song_name = item_name.split("#")
            if singer in hparams['short_datasets']:
                args.append([singer, r])
        list(tqdm(chunked_multiprocess_run(self.save_short_job, args, ordered=False), total=len(args),
                  desc='Saving phoneme for short datasets.'))

    @staticmethod
    def save_short_job(singer, r):
        wav_fn = r['wav_fn']
        seg_out_dir = f"{hparams['seg_out_dir']}/{singer}"
        os.makedirs(seg_out_dir, exist_ok=True)
        with open(f'{seg_out_dir}/{os.path.basename(wav_fn)[:-4]}.txt', 'w') as f:
            f.write(r['txt'])
        subprocess.check_call(f'cp "{wav_fn}" "{seg_out_dir}/"', shell=True)

    @property
    def train_item_names(self):
        return self.item_names

    @property
    def valid_item_names(self):
        return self.item_names[:5]

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
            wav_fn = self.item2wavfn[item_name]
            yield item_name, ph, txt, wav_fn

    @classmethod
    def process_item(cls, item_name, ph, txt, wav_fn, encoder, binarization_args):
        wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_fn)
        try:
            phone_encoded = encoder.encode(ph)
        except:
            traceback.print_exc()
            print(f"| [Empty phoneme] Skip {item_name}, {wav_fn}")
            return None
        res = {
            'phone': phone_encoded, 'txt': txt, 'ph': ph, 'mel': mel, 'wav': wav,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0],
        }
        return res


if __name__ == "__main__":
    SingingPreSegPreAlign().process()
    SingingPreSegBinarizer().process()
