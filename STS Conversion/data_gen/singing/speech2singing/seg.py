# %%
import os
import multiprocessing as mp
import copy
import sys

from utils.hparams import hparams, set_hparams
from utils import audio
from usr.usr_utils import *

import textgrid
from tqdm import tqdm
import librosa
from dtw import dtw
import numpy as np
import pandas as pd

# %%
class Seg:
    def __init__(self):
        set_hparams()
        self.raw_data_dir = hparams["raw_data_dir"]
        self.processed_long_data_dir = hparams["processed_long_data_dir"]
        self.processed_short_data_dir = hparams["processed_short_data_dir"]

        self.meta = []
        self.get_meta()

    def get_meta(self):
        for spk in os.listdir(f"{self.raw_data_dir}"):
            for f_name in os.listdir(f"{self.raw_data_dir}/{spk}"):
                if f_name[-14:] == '-corrected.txt' and 'agg' not in f_name:
                    item_name = f_name[:-14]
                    if os.path.exists(f"{self.raw_data_dir}/{spk}/{item_name} 歌词.wav") and \
                            os.path.exists(f"{self.raw_data_dir}/{spk}/{item_name} 专业.wav") and \
                            os.path.exists(
                                f"{self.processed_long_data_dir}/mfa_outputs/{spk}#{item_name} 歌词.TextGrid") and \
                            os.path.exists(f"{self.processed_long_data_dir}/mfa_outputs/{spk}#{item_name} 专业.TextGrid"):
                        item1 = {
                            "textgrid": f"{self.processed_long_data_dir}/mfa_outputs/{spk}#{item_name} 专业.TextGrid",
                            "wav": f"{self.raw_data_dir}/{spk}/{item_name} 专业.wav",
                            "text": f"{self.raw_data_dir}/{spk}/{f_name}",
                            "item_name": item_name,
                            "type": "专业",
                            "spk": spk
                        }
                        item2 = {
                            "textgrid": f"{self.processed_long_data_dir}/mfa_outputs/{spk}#{item_name} 歌词.TextGrid",
                            "wav": f"{self.raw_data_dir}/{spk}/{item_name} 歌词.wav",
                            "text": f"{self.raw_data_dir}/{spk}/{f_name}",
                            "item_name": item_name,
                            "type": "歌词",
                            "spk": spk
                        }
                        self.meta.extend([item1, item2])
                    if os.path.exists(f"{self.raw_data_dir}/{spk}/{item_name} 业余.wav") and \
                            os.path.exists(f"{self.processed_long_data_dir}/mfa_outputs/{spk}#{item_name} 业余.TextGrid"):
                        item3 = {
                            "textgrid": f"{self.processed_long_data_dir}/mfa_outputs/{spk}#{item_name} 业余.TextGrid",
                            "wav": f"{self.raw_data_dir}/{spk}/{item_name} 业余.wav",
                            "text": f"{self.raw_data_dir}/{spk}/{f_name}",
                            "item_name": item_name,
                            "type": "业余",
                            "spk": spk
                        }
                        self.meta.append(item3)
        self.meta.sort(key=lambda x: f"{x['spk']}#{x['item_name']}")

    def _get_interval(self, tg, text):
        # split according to word boundary and eliminate ill symbols
        for i in range(len(text)):
            text[i] = re.sub(r"[^a-zA-Z0-9 '\",.]", "", text[i])
        text = [line.split() for line in text]
        words_itv, phones_itv = (tg.tiers[0].intervals, tg.tiers[1].intervals) if tg.tiers[0].name == "words" \
            else (tg.tiers[1].intervals, tg.tiers[0].intervals)
        txt_sen_idx, txt_begin_idx, txt_end_idx, txt_word_idx = 0, 0, 0, 0  # 句子的idx，txt中每句话开始和结束的idx，txt每句话的word的idx
        tg_word_idx, tg_phone_idx = 0, 0    # tg中word和phone的idx
        wav_intervals = []
        sen_begin, sen_end = 0, 0
        word_idx_intervals = []
        word_idx_begin, word_idx_end = 0, 0     # inclusive
        # loop for words
        while True:
            if tg_word_idx == len(words_itv) or txt_sen_idx == len(text):
                break
            if words_itv[tg_word_idx].mark == "" or words_itv[tg_word_idx].mark.lower() == "sil":
                tg_word_idx += 1
                continue
            if txt_word_idx == 0:
                sen_begin = words_itv[tg_word_idx].minTime
                word_idx_begin = tg_word_idx
            # 处理数字的异常
            if text[txt_sen_idx][txt_word_idx] == "24" or text[txt_sen_idx][txt_word_idx] == "29th":
                tg_word_idx += 1    # twenty four, 两个字
            if text[txt_sen_idx][txt_word_idx] == "a.m.":
                tg_word_idx += 2    # a.m.被分开发音了，而且中间有sil
            if text[txt_sen_idx][txt_word_idx] == "1234":
                txt_word_idx = 0
                txt_sen_idx += 2
                tg_word_idx = 21 - 1
                wav_intervals.extend([None, None])
                word_idx_intervals.extend([None, None])
                continue
            # 处理句末
            if txt_word_idx == len(text[txt_sen_idx]) - 1:
                sen_end = words_itv[tg_word_idx].maxTime
                word_idx_end = tg_word_idx  # inclusive
                txt_word_idx = 0
                txt_sen_idx += 1
                tg_word_idx += 1
                wav_intervals.append([sen_begin, sen_end])
                word_idx_intervals.append([word_idx_begin, word_idx_end])
                continue
            txt_word_idx += 1
            tg_word_idx += 1
        # loop for phones
        phone_idx_intervals = []
        for i, wav_interval in enumerate(wav_intervals):
            if wav_interval is None:
                phone_idx_intervals.append(None)    # 同步
                continue
            sen_begin, sen_end = wav_interval
            # find beginning
            while phones_itv[tg_phone_idx].minTime < sen_begin:
                tg_phone_idx += 1
            assert np.isclose(phones_itv[tg_phone_idx].minTime, sen_begin)
            phone_idx_begin = tg_phone_idx
            # find ending
            while phones_itv[tg_phone_idx].maxTime < sen_end:
                tg_phone_idx += 1
            assert np.isclose(phones_itv[tg_phone_idx].maxTime, sen_end)
            phone_idx_end = tg_phone_idx
            # add
            phone_idx_intervals.append([phone_idx_begin, phone_idx_end])

        return wav_intervals, word_idx_intervals, phone_idx_intervals

    def process_one(self, item):
        item_name = item["item_name"]
        spk = item["spk"]
        song_type = item["type"]

        # read text
        with open(item["text"]) as f:
            text = f.read()
        text = [line.strip() for line in text.strip().split("\n")]

        # read textgrid
        tg = textgrid.TextGrid()
        tg.read(item["textgrid"])
        words_itv, phones_itv = (tg.tiers[0].intervals, tg.tiers[1].intervals) if tg.tiers[0].name == "words" \
            else (tg.tiers[1].intervals, tg.tiers[0].intervals)

        # read wav
        wav, sr = librosa.core.load(item["wav"], sr=None)

        # get intervals
        try:
            wav_intervals, word_idx_intervals, phone_idx_intervals = self._get_interval(tg, text)
        except IndexError as err:
            print(err)
            print(item)
            sys.exit()

        mkdir(f"{self.processed_short_data_dir}")
        mkdir(f"{self.processed_short_data_dir}/wav")
        mkdir(f"{self.processed_short_data_dir}/textgrid")
        mkdir(f"{self.processed_short_data_dir}/text")
        mkdir(f"{self.processed_short_data_dir}/wav/{spk}#{item_name}#{song_type}")
        mkdir(f"{self.processed_short_data_dir}/textgrid/{spk}#{item_name}#{song_type}")
        mkdir(f"{self.processed_short_data_dir}/text/{spk}#{item_name}#{song_type}")
        for idx, (wav_interval, word_idx_interval, phone_idx_interval) \
                in enumerate(zip(wav_intervals, word_idx_intervals, phone_idx_intervals)):
            # save wav
            if wav_interval is None or word_idx_interval is None or phone_idx_interval is None:
                continue
            begin_frame = int(sr * wav_interval[0])
            end_frame = int(sr * wav_interval[1])
            wav_out = wav[begin_frame: end_frame]
            audio.save_wav(wav_out, f"{self.processed_short_data_dir}/wav/{spk}#{item_name}#{song_type}/{spk}#{item_name}#{song_type}#{idx}.wav", sr)

            # save text
            with open(f"{self.processed_short_data_dir}/text/{spk}#{item_name}#{song_type}/{spk}#{item_name}#{song_type}#{idx}.txt", "w") as f:
                f.write(text[idx])

            # save textgrid
            tg_sen = textgrid.TextGrid(minTime=0., maxTime=0.)
            tier_word = textgrid.IntervalTier(name="words", minTime=0., maxTime=0.)  # 添加一层,命名为words层
            tier_phone = textgrid.IntervalTier(name="phones", minTime=0., maxTime=0.)  # 添加一层,命名为phones层
            offset = words_itv[word_idx_interval[0]].minTime
            for tg_idx in range(word_idx_interval[0], word_idx_interval[1]+1):
                interval = textgrid.Interval(minTime=round(words_itv[tg_idx].minTime-offset, 2),
                                             maxTime=round(words_itv[tg_idx].maxTime-offset, 2),
                                             mark=words_itv[tg_idx].mark)
                tier_word.addInterval(interval)
            for tg_idx in range(phone_idx_interval[0], phone_idx_interval[1]+1):
                interval = textgrid.Interval(minTime=round(phones_itv[tg_idx].minTime-offset, 2),
                                             maxTime=round(phones_itv[tg_idx].maxTime-offset, 2),
                                             mark=phones_itv[tg_idx].mark)
                tier_phone.addInterval(interval)
            tg_sen.tiers.append(tier_word)
            tg_sen.tiers.append(tier_phone)
            tg_sen.maxTime = tier_word.maxTime = tier_phone.maxTime = \
                round(float(words_itv[word_idx_interval[1]].maxTime - offset), 2)
            tg_sen.write(f"{self.processed_short_data_dir}/textgrid/{spk}#{item_name}#{song_type}/{spk}#{item_name}#{song_type}#{idx}.TextGrid")

    def process(self):
        for idx, item in tqdm(enumerate(self.meta), desc="seg", ncols=80, total=len(self.meta)):
            self.process_one(item)

    def seg_meta(self):
        # 将 metadata_phone 文件切分并预处理
        # meta_df = pd.read_csv(f"{self.processed_long_data_dir}/metadata_phone.csv")
        new_df = pd.DataFrame(columns=["item_name", "spk", "txt", "txt_raw", "ph", "speech_wav_fn", "sing_wav_fn"])
        for speech_wav_dir in tqdm(sorted(os.listdir(f"{self.processed_short_data_dir}/wav/")), total=550, desc="seg meta", ncols=80):
            spk, song_name, song_type = speech_wav_dir.split("#")
            if song_type == "歌词" and os.path.exists(f"{self.processed_short_data_dir}/wav/{spk}#{song_name}#专业"):
                for speech_wav_fn in sorted(os.listdir(f"{self.processed_short_data_dir}/wav/{speech_wav_dir}/")):
                    spk, song_name, song_type, sen_id = speech_wav_fn[:-4].split("#")
                    if os.path.exists(f"{self.processed_short_data_dir}/wav/{spk}#{song_name}#专业/{spk}#{song_name}#专业#{sen_id}.wav"):
                        new_item = {"item_name": f"{spk}#{song_name}#{sen_id}", "spk": spk}
                        with open(f"{self.processed_short_data_dir}/text/{spk}#{song_name}#专业/{spk}#{song_name}#专业#{sen_id}.txt") as f:
                            new_item["txt_raw"] = f.read().strip()
                            new_item["txt"] = new_item["txt_raw"].lower()
                        tg = textgrid.TextGrid()
                        tg.read(f"{self.processed_short_data_dir}/textgrid/{spk}#{song_name}#专业/{spk}#{song_name}#专业#{sen_id}.TextGrid")
                        words_itv, phones_itv = (tg.tiers[0].intervals, tg.tiers[1].intervals) if tg.tiers[0].name == "words" \
                            else (tg.tiers[1].intervals, tg.tiers[0].intervals)
                        ph_seg = "<BOS>"
                        for word_idx, word in enumerate(words_itv):
                            if word.mark == "" or word.mark.lower() == "sil":
                                continue
                            if word_idx == 0:
                                ph_seg = ph_seg + " " + word.mark.upper().replace("_", " ")
                            else:
                                ph_seg = ph_seg + " | " + word.mark.upper().replace("_", " ")
                        ph_seg = ph_seg + " <EOS>"
                        new_item["ph"] = ph_seg
                        new_item["speech_wav_fn"] = f"{self.processed_short_data_dir}/wav/{speech_wav_dir}/{speech_wav_fn}"
                        new_item["sing_wav_fn"] = f"{self.processed_short_data_dir}/wav/{spk}#{song_name}#专业/{spk}#{song_name}#专业#{sen_id}.wav"
                        new_df = new_df.append(new_item, ignore_index=True)
        new_df.to_csv(f"{self.processed_short_data_dir}/metadata_phone.csv")

    def after_process(self):
        # copy the meta_phone.csv file to the new processed_dir
        # exe_cmd(f"cp {self.processed_long_data_dir}/metadata_phone.csv {self.processed_short_data_dir}/")
        # exe_cmd(f"cp -r {self.processed_long_data_dir}/mfa_outputs {self.processed_short_data_dir}/")
        self.seg_meta()
        # copy the dict.txt file to the new processed_dir
        exe_cmd(f"cp {self.processed_long_data_dir}/dict.txt {self.processed_short_data_dir}/")


# %%
if __name__ == '__main__':
    seg = Seg()
    print()
    seg.process()
    seg.after_process()

