# preprocess opensvi 数据
import os
import textgrid
import json
from resemblyzer import VoiceEncoder
from tqdm import tqdm
import glob
import numpy as np
import librosa
from utils.text.text_encoder import build_token_encoder
import re

all_data_dir = "/data_disk/opensvi_data"

processed_data_dir = "/home/renyi/hjz/NATSpeech/data/processed/opensvi"
ph_encoder = build_token_encoder(os.path.join(processed_data_dir, "phone_set.json"))

ALL_SHENGMU = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'x', 'z', 'zh']
ALL_YUNMU = ['a', 'ai', 'an', 'ang', 'ao',  'e', 'ei', 'en', 'eng', 'er',  'i', 'ia', 'ian', 'iang', 'iao',
             'ie', 'in', 'ing', 'iong', 'iou', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'uei',
             'uen', 'uo', 'v', 'van', 've', 'vn']

def tg2text(tg_path):
    tg = textgrid.TextGrid.fromFile(tg_path)
    word_intervals = tg[0]
    ph_intervals = tg[1]
    base_name = os.path.basename(tg_path)
    word_name = base_name[:-9] + "_word" + ".txt"
    ph_name = base_name[:-9] + '_ph' + '.txt'
    dir_path = os.path.dirname(tg_path)
    with open(os.path.join(dir_path, word_name), 'w') as f:
        for word in word_intervals:
            mark = word.mark
            # mark = mark.strip()
            if mark is None or mark == '':
                mark = "_NONE"
            line = mark + ' || ' + str(word.minTime) + ' || ' + str(word.maxTime) + '\n'
            f.write(line)
    idx = 0
    with open(os.path.join(dir_path, ph_name), 'w') as f:
        while idx < len(ph_intervals):
            ph = ph_intervals[idx]
            mark = ph.mark
            if mark is None or mark == '':
                mark = "_NONE"
                line = mark + ' || ' + str(ph.minTime) + ' || ' + str(ph.maxTime) + '\n'
                idx = idx + 1
            elif mark == "w":
                if ph_intervals[idx+1].mark == "u":
                    new_mark = ph_intervals[idx+1].mark
                else:
                    new_mark = "u" + ph_intervals[idx+1].mark
                line = new_mark + ' || ' + str(ph.minTime) + ' || ' + str(ph_intervals[idx+1].maxTime) + '\n'
                idx = idx + 2
            elif mark == "y":
                if ph_intervals[idx+1].mark in ["a", "an", "ang", "ao", "e", "o", "ong", "ou"]:
                    new_mark = "i" + ph_intervals[idx+1].mark
                elif ph_intervals[idx+1].mark in ["i", "in", "ing", "v", "van", "ve", "vn"]:
                    new_mark = ph_intervals[idx+1].mark
                elif ph_intervals[idx+1].mark == "E":
                    new_mark = "ie"
                elif ph_intervals[idx+1].mark == "En":
                    new_mark = "ian"
                else:
                    assert False, "error match"
                line = new_mark + ' || ' + str(ph.minTime) + ' || ' + str(ph_intervals[idx+1].maxTime) + '\n'
                idx = idx + 2
            elif mark in ["ir", "i0"]:
                new_mark = "i"
                line = new_mark + ' || ' + str(ph.minTime) + ' || ' + str(ph.maxTime) + '\n'
                idx = idx + 1
            elif mark == "SP":
                new_mark = "<SP>"
                line = new_mark + ' || ' + str(ph.minTime) + ' || ' + str(ph.maxTime) + '\n'
                idx = idx + 1
            elif mark == "AP":
                new_mark = "<AP>"
                line = new_mark + ' || ' + str(ph.minTime) + ' || ' + str(ph.maxTime) + '\n'
                idx = idx + 1
            else:
                line = mark + ' || ' + str(ph.minTime) + ' || ' + str(ph.maxTime) + '\n'
                idx = idx + 1
            f.write(line)

# 处理textgrid 到 json，处理word 与 ph 之间的对应关系
def tg_hier(tg_path):
    word_dict_list = []
    # 这里利用txt 文本是为了方便修改，同时也不破坏原始数据
    word_path = tg_path.replace('.TextGrid', '_word.txt')
    ph_path = tg_path.replace('.TextGrid', '_ph.txt')
    with open(word_path, 'r') as f:
        word_list = f.readlines()
    with open(ph_path, 'r') as f:
        ph_list = f.readlines()

    # with open("log.txt", 'w') as f:
    for word in word_list:
        word = word.strip()
        # f.write(str(word) + '\n')
        word_dict = {}
        mark, min, max = word.split(' || ')
        word_dict["word"] = mark
        word_dict["start_time"] = start_time = min
        word_dict["end_time"] = end_time = max
        word_dict_list.append(word_dict)
    idx = 0
    with open("log.txt", 'w') as f:
        for ph in ph_list:
            ph = ph.strip()
            mark, min, max = ph.split(' || ')
            f.write(str(ph))
            if mark in ALL_SHENGMU:
                word_dict_list[idx]["shengmu"] = mark
                word_dict_list[idx]["sm_start"] = min
                word_dict_list[idx]["sm_end"] = max
                f.write(word_dict_list[idx]["word"] + '\n')
            else:
                word_dict_list[idx]["yunmu"] = mark
                word_dict_list[idx]["ym_start"] = min
                word_dict_list[idx]["ym_end"] = max
                f.write(word_dict_list[idx]["word"] + '\n')
                idx = idx + 1

    assert idx == len(word_dict_list), "error"
    with open(tg_path.replace(f".TextGrid", "_tg.json"), 'w', encoding="utf8") as f:
        json.dump(word_dict_list, f, indent=4, ensure_ascii=False)


def preproces():
    meta_dicts = []
    for tg_json in tqdm(glob.glob(os.path.join(all_data_dir, "*", "*", "*_tg.json"))):
        meta_dict = {}
        item_name = "#".join(tg_json.split("/")[-3:])[:-8].replace("textgrids#", "")
        meta_dict["item_name"] = item_name
        tg_dicts = json.load(open(tg_json))
        phs = []
        ph_durs = []
        word_id = 1
        ph2word = []
        for tg_dict in tg_dicts:
            if "shengmu" in tg_dict:
                phs.append(tg_dict["shengmu"])
                ph_durs.append(float(tg_dict["sm_end"]) - float(tg_dict["sm_start"]))
                ph2word.append(word_id)
            phs.append(tg_dict["yunmu"])
            ph_durs.append(float(tg_dict["ym_end"]) - float(tg_dict["ym_start"]))
            ph2word.append(word_id)
            word_id = word_id + 1
        meta_dict["ph"] = " ".join(phs)
        meta_dict["ph_token"] = ph_encoder.encode(meta_dict["ph"])
        meta_dict["ph2word"] = ph2word
        meta_dict["ph_durs"] = ph_durs
        meta_dict["wav_fn"] = tg_json.replace("textgrids/", "").replace("_tg.json", ".wav")
        meta_dict["spk_fn"] = meta_dict["wav_fn"].replace(".wav", "_spk.npy")
        meta_dicts.append(meta_dict)
    with open(f"{processed_data_dir}/metadata.json", 'w') as f:
        f.write(re.sub(r'\n\s+([\d+\]])', r'\1', json.dumps(meta_dicts, ensure_ascii=False, sort_keys=False, indent=1)))




# opensvi 和 我们的ph 标注有的是不对应的
def tg2json():
    for tg_path in tqdm(glob.glob(os.path.join(all_data_dir, "*", "*", "*.TextGrid"))):
        # item_name = "#".json(tg_path.split("/")[-3:])
        tg2text(tg_path)
        tg_hier(tg_path)


# spker extractor
def spker_extractor(audio_dir):
    ve = VoiceEncoder()
    for wav_fn in tqdm(glob.glob(os.path.join(audio_dir, "*/*.wav"))):
        wav_data, sample_rate = librosa.load(wav_fn, 24000)
        spk_embed = ve.embed_utterance(wav_data.astype(float))
        spk_fn = wav_fn.replace(".wav", "_spk.npy")
        np.save(spk_fn, spk_embed)


if __name__ == "__main__":
    tg2json()
    # spker_extractor(all_data_dir)
    preproces()
    
    