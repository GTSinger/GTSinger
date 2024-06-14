# xiaoma 数据处理
from resemblyzer import VoiceEncoder
from tqdm import tqdm
import glob
import numpy as np
import os
import librosa
import textgrid
from tqdm import tqdm
from utils.text.text_encoder import build_token_encoder
import re
import json

ALL_SHENGMU = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'x', 'z', 'zh']
ALL_YUNMU = ['a', 'ai', 'an', 'ang', 'ao',  'e', 'ei', 'en', 'eng', 'er',  'i', 'ia', 'ian', 'iang', 'iao',
             'ie', 'in', 'ing', 'iong', 'iou', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'uei',
             'uen', 'uo', 'v', 'van', 've', 'vn']

processed_data_dir = "/home/renyi/hjz/NATSpeech/data/processed/xiaoma"
ph_encoder = build_token_encoder(os.path.join(processed_data_dir, "phone_set.json"))

def extract_word(phs):
    ph2word = []
    word_id = 1
    words = []
    current_word = ""
    for ph in phs:
        if ph in ALL_SHENGMU:
            ph2word.append(word_id)
            current_word = ph
        else:
            ph2word.append(word_id)
            if current_word == "":
                current_word = ph
            else:
                current_word = current_word + "_" + ph
            words.append(current_word)
            word_id = word_id + 1
            current_word = ""
    return ph2word, words

# spker extractor
def spker_extractor(audio_dir):
    ve = VoiceEncoder()
    for wav_fn in tqdm(glob.glob(os.path.join(audio_dir, "*/*_wf0.wav"))):
        wav_data, sample_rate = librosa.load(wav_fn, 24000)
        spk_embed = ve.embed_utterance(wav_data.astype(float))
        spk_fn = wav_fn.replace(".wav", "_spk.npy")
        np.save(spk_fn, spk_embed)

def norm_txt(txt_dir):
    def is_sil_phoneme(p):
        return not p[0].isalpha()

    for ph_fn in tqdm(glob.glob(os.path.join(txt_dir, "*/*_ph.txt"))):
        tg_fn = ph_fn.replace("_ph.txt", ".TextGrid")
        with open(ph_fn, "r") as f:
            ph_list = f.readline().strip().split(" ")
        tg = textgrid.TextGrid.fromFile(tg_fn)
        ph_idx = 0
        tg_idx = 0
        target_ph_list = []
        while tg_idx < len(tg[1]):
            t = tg[1][tg_idx]
            if t.mark == "" and is_sil_phoneme(ph_list[ph_idx]):
                target_ph_list.append("<SP>")
                tg_idx = tg_idx + 1
                ph_idx = ph_idx + 1
            elif t.mark != "" and is_sil_phoneme(ph_list[ph_idx]):
                ph_idx = ph_idx + 1
                continue
            elif t.mark != "" and not is_sil_phoneme(ph_list[ph_idx]) and t.mark == ph_list[ph_idx]:
                target_ph_list.append(ph_list[ph_idx])
                tg_idx = tg_idx + 1
                ph_idx = ph_idx + 1
            else:
                print(t.mark)
                print(ph_list[ph_idx])
                assert False, "error match"
        target_phs = " ".join(target_ph_list)
        norm_ph_fn = ph_fn.replace("_ph.txt", "_normph.txt")
        with open(norm_ph_fn, "w") as f:
            f.write(target_phs)

def preprocess(data_dir):
    meta_dicts = []
    for wav_fn in tqdm(glob.glob(os.path.join(data_dir, "*/*_wf0.wav"))):
        meta_dict = {}
        song_name = wav_fn.split("/")[-2]
        split = wav_fn.split("/")[-1].replace("_wf0.wav", "")
        meta_dict["item_name"] = f"{song_name}#{split}"
        meta_dict["wav_fn"] = wav_fn
        meta_dict["spk_fn"] = wav_fn.replace("_wf0.wav", "_spk.npy")
        meta_dict["ph"] = open(wav_fn.replace("_wf0.wav", "_normph.txt")).readline()
        meta_dict["ph_token"] = ph_encoder.encode(meta_dict["ph"])
        meta_dict["ph2word"], meta_dict["word_token"] = extract_word(meta_dict["ph"].split(" "))
        meta_dict["tg_fn"] = wav_fn.replace("_wf0.wav", ".TextGrid")
        meta_dicts.append(meta_dict)
    with open(f"{processed_data_dir}/metadata.json", 'w') as f:
            f.write(re.sub(r'\n\s+([\d+\]])', r'\1', json.dumps(meta_dicts, ensure_ascii=False, sort_keys=False, indent=1)))

if __name__ == "__main__":
    # spker_extractor("/data_disk/xiaoma/splits")
    # norm_txt("/data_disk/xiaoma/splits")
    preprocess("/data_disk/xiaoma/splits")