#-*- coding : utf-8-*-
# coding:unicode_escape
import os
from data_gen.tts.base_pre_align import BasePreAlign
from utils.hparams import hparams
import glob


class Speech2SingingPreAlign(BasePreAlign):
    def meta_data(self):
        for dataset in hparams['datasets']:
            wav_fns = sorted(
                glob.glob(f'{self.raw_data_dir}/{dataset}*/*.mp3') +
                glob.glob(f'{self.raw_data_dir}/{dataset}*/*/*.mp3') +
                glob.glob(f'{self.raw_data_dir}/{dataset}*/*.wav') +
                glob.glob(f'{self.raw_data_dir}/{dataset}*/*/*.wav')
            )
            print(f"| {dataset}: {len(wav_fns)}")
            for wav_fn in wav_fns:
                item_name = dataset + "#" + os.path.basename(wav_fn)[:-4]
                txt_fn = (wav_fn[:-6]).strip() + '-agg-corrected.txt'     # 去掉“专业”，“歌词”等词，加上"-agg-corrected"
                yield item_name, wav_fn, (self.load_txt, txt_fn), dataset


if __name__ == "__main__":
    Speech2SingingPreAlign().process()
