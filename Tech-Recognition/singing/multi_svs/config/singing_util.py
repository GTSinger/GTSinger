import miditoolkit
import os
from tqdm import tqdm
from utils.commons.indexed_datasets import IndexedDatasetBuilder, IndexedDataset
import numpy as np
import json
import numpy as np

ALL_SHENMU = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'x', 'z', 'zh']
ALL_YUNMU = ['a', 'ai', 'an', 'ang', 'ao',  'e', 'ei', 'en', 'eng', 'er',  'i', 'ia', 'ian', 'iang', 'iao',
             'ie', 'in', 'ing', 'iong', 'iou', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'uei',
             'uen', 'uo', 'v', 'van', 've', 'vn']

def dataset_mix(prefix):
    data_dir = "/home/renyi/hjz/NATSpeech/data/binary/m4_zhengshu"
    data_dir1 = "/home/renyi/hjz/NATSpeech/data/binary/xiaoma"
    data_dir2 = "/home/renyi/hjz/NATSpeech/data/binary/opensvi"
    mix_data_dir = "data/binary/xiaoma_m4_opensvi"
    os.makedirs(mix_data_dir, exist_ok=True)
    builder = IndexedDatasetBuilder(f'{mix_data_dir}/{prefix}')
    lengths = []
    total_sec = 0
    indexed_ds = tqdm(IndexedDataset(f'{data_dir}/{prefix}'))
    for item in indexed_ds:
        builder.add_item(item)
        lengths.append(item['len'])
        total_sec += item['sec']

    indexed_ds = tqdm(IndexedDataset(f'{data_dir1}/{prefix}'))
    for item in indexed_ds:
        builder.add_item(item)
        lengths.append(item['len'])
        total_sec += item['sec']

    indexed_ds = tqdm(IndexedDataset(f'{data_dir2}/{prefix}'))
    for item in indexed_ds:
        builder.add_item(item)
        lengths.append(item['len'])
        total_sec += item['sec']
        
    builder.finalize()
    np.save(f'{mix_data_dir}/{prefix}_lengths.npy', lengths)
    print(f"| {prefix} total duration: {total_sec:.3f}s")

def extract_word(phs, mel2ph):
    ph2word = []
    word_id = 1
    for ph in phs:
        if ph in ALL_SHENMU:
            ph2word.append(word_id)
        else:
            ph2word.append(word_id)
            word_id = word_id + 1
    mel2word = []
    dur_word = [0 for _ in range(max(ph2word) + 1)]
    for i, m2p in enumerate(mel2ph):
        word_idx = ph2word[m2p - 1]
        mel2word.append(ph2word[m2p - 1])
        dur_word[word_idx] += 1
    return ph2word, mel2word

def add_word(prefix):
    data_dir = "data/binary/xiaoma_m4"
    target_data_dir = "data/binary/xiaoma_m4_1"
    os.makedirs(target_data_dir, exist_ok=True)
    builder = IndexedDatasetBuilder(f'{target_data_dir}/{prefix}')
    lengths = []
    f0s = []
    total_sec = 0
    indexed_ds = tqdm(IndexedDataset(f'{data_dir}/{prefix}'))
    for item in indexed_ds:
        ph = item["ph"]
        mel2ph = item["mel2ph"]
        # try:
        ph2word, mel2word = extract_word(ph.split(" "), mel2ph)
        item["ph2word"] = ph2word
        item["mel2word"] = mel2word
        builder.add_item(item)
        lengths.append(item['len'])
        total_sec += item['sec']
        if item.get('f0') is not None:
            f0s.append(item['f0'])
        # except:
            

    builder.finalize()
    np.save(f'{target_data_dir}/{prefix}_lengths.npy', lengths)
    if len(f0s) > 0:
        f0s = np.concatenate(f0s, 0)
        f0s = f0s[f0s != 0]
        np.save(f'{target_data_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
    print(f"| {prefix} total duration: {total_sec:.3f}s")

def get_name2cleanph(prefix):
    data_dir = "/home/renyi/hjz/NATSpeech/data/binary/m4_zhengshu1"
    indexed_ds = tqdm(IndexedDataset(f'{data_dir}/{prefix}'))
    metas = json.load(open("/data_disk/singing-process/m4singer-zhengshu/meta.json"))
    item_name2id = {}
    for item in indexed_ds:
        item_name = item["item_name"]
        for i, meta in enumerate(metas):
            if item_name == meta["item_name"]:
                item_name2id[item_name] = meta["phs"]
                break
    with open('name2ph.json', 'w', encoding="utf8") as f:
        json.dump(item_name2id, f, ensure_ascii=False)

def del_wav(prefix):
    data_dir = "data/binary/xiaoma_m4"
    tgt_data_dir = "data/binary/xiaoma_m4_1"
    os.makedirs(tgt_data_dir, exist_ok=True)
    builder = IndexedDatasetBuilder(f'{tgt_data_dir}/{prefix}')
    lengths = []
    f0s = []
    total_sec = 0
    indexed_ds = tqdm(IndexedDataset(f'{data_dir}/{prefix}'))
    for item in indexed_ds:
        del item["wav"]
        builder.add_item(item)
        lengths.append(item['len'])
        total_sec += item['sec']
        if item.get('f0') is not None:
            f0s.append(item['f0'])
    builder.finalize()
    np.save(f'{tgt_data_dir}/{prefix}_lengths.npy', lengths)
    if len(f0s) > 0:
        f0s = np.concatenate(f0s, 0)
        f0s = f0s[f0s != 0]
        np.save(f'{tgt_data_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
    print(f"| {prefix} total duration: {total_sec:.3f}s")


        

if __name__ == "__main__":
    # midi_fn = "/data_disk/xiaoma/splits/0514#女_2_4-像鱼/0000.mid"
    # mf = miditoolkit.MidiFile(midi_fn)
    # instru = mf.instruments[0]

    # notes = instru.notes

    # print(notes[0])
    # dataset_mix("valid")
    # add_word("train")
    # get_name2cleanph("test")
    # del_wav("train")
    # del_wav("test")
    # del_wav("valid")
    # dataset_mix("valid")
    # dataset_mix("test")
    # dataset_mix("train")
    lengths = np.load("/home/renyi/hjz/NATSpeech/data/binary/xiaoma/train_lengths.npy")
    cnt = 1
    all_lengths = 0
    long_lengths = 0
    for length in lengths:
        if length > 4000:
            long_lengths = long_lengths + length
        else:
            all_lengths = all_lengths + length
    print(long_lengths / all_lengths)
    # print(max(lengths))
