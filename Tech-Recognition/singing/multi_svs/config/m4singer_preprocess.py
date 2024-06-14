# m4singer 将slur repeat 的 ph 删除，同时AP SP --> SP
# 提取 spk emb
import json
import os
from tqdm import tqdm
import glob
import numpy as np
from resemblyzer import VoiceEncoder
import librosa
from utils.text.text_encoder import build_token_encoder
import re

# ve = VoiceEncoder()
processed_data_dir = "/home/renyi/hjz/NATSpeech/data/processed/m4_zhengshu"
ph_encoder = build_token_encoder(os.path.join(processed_data_dir, "phone_set.json"))


ALL_SHENGMU = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'x', 'z', 'zh']
ALL_YUNMU = ['a', 'ai', 'an', 'ang', 'ao',  'e', 'ei', 'en', 'eng', 'er',  'i', 'ia', 'ian', 'iang', 'iao',
             'ie', 'in', 'ing', 'iong', 'iou', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'uei',
             'uen', 'uo', 'v', 'van', 've', 'vn']
    

raw_data_dir = '/data_disk/singing-process/m4singer-zhengshu'
def preprocess():
    song_items = json.load(open(os.path.join(raw_data_dir, 'meta.json')))  # [list of dict]
    new_dicts = []
    for song_item in tqdm(song_items):
        phs = song_item["phs"]
        tgt_phs = [] 
        is_slur = song_item["is_slur"]
        # 替换 <AP>，删除 slur
        for ph, s in zip(phs, is_slur):
            # if ph == "<AP>" or ph == "<SP>":
            #     tgt_phs.append("<SP>")
            if s == 0:
                tgt_phs.append(ph)
        # 合并slur 时长
        ph_dur = song_item["ph_dur"]
        tgt_ph_dur = []
        for d, s in zip(ph_dur, is_slur):
            if s == 0:
                tgt_ph_dur.append(d)
            else:
                tgt_ph_dur[-1] = tgt_ph_dur[-1] + d
        assert len(tgt_phs) == len(tgt_ph_dur), "error match 1"
        assert abs(sum(ph_dur) - sum(tgt_ph_dur)) < 0.000001, f"time loss 1 {sum(ph_dur)}, {sum(tgt_ph_dur)}"
        # 合并重复的<SP>
        # tgt_phs1 = []
        # tgt_ph_dur1 = []
        # for ph, d in zip(tgt_phs, tgt_ph_dur):
        #     if ph == "<SP>" and len(tgt_phs1) > 0 and tgt_phs1[-1] == "<SP>":
        #         tgt_ph_dur1[-1] = tgt_ph_dur1[-1] + d
        #     else:
        #         tgt_phs1.append(ph)
        #         tgt_ph_dur1.append(d)
        # assert len(tgt_phs1) == len(tgt_ph_dur1), "error match 2"
        # assert abs(sum(ph_dur) - sum(tgt_ph_dur1)) < 0.000001, "time loss 2"
        # ph2word 构建
        ph2word = []
        word_id = 0
        for ph in tgt_phs:
            if ph in ALL_SHENGMU:
                ph2word.append(word_id)
            else:
                ph2word.append(word_id)
                word_id = word_id + 1
        
        song_item["ph"] = " ".join(tgt_phs)
        song_item["ph_durs"] = tgt_ph_dur
        song_item["ph2word"] = ph2word
        song_item["ph_token"] = ph_encoder.encode(song_item["ph"])
        singer, song_name, sent_id = song_item["item_name"].split("#")
        song_item["wav_fn"] = f'{raw_data_dir}/{singer}#{song_name}/{sent_id}.wav'
        song_item["spk_fn"] = song_item["wav_fn"].replace(".wav", "_spk.npy")
        new_dicts.append(song_item)
    with open(f"{processed_data_dir}/metadata.json", 'w') as f:
        f.write(re.sub(r'\n\s+([\d+\]])', r'\1', json.dumps(new_dicts, ensure_ascii=False, sort_keys=False, indent=1)))

def spker_extractor(audio_dir):
    for wav_fn in tqdm(glob.glob(os.path.join(audio_dir, "*/*.wav"))):
        wav_data, sample_rate = librosa.load(wav_fn, 24000)
        spk_embed = ve.embed_utterance(wav_data.astype(float))
        spk_fn = wav_fn.replace(".wav", "_spk.npy")
        np.save(spk_fn, spk_embed)


if __name__ == "__main__":
    # spker_extractor(raw_data_dir)
    preprocess()