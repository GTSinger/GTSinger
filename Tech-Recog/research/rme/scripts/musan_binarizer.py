# %%
import os
import glob
from pathlib import Path

from tqdm import tqdm
import numpy as np
import librosa
from utils.commons.indexed_datasets import IndexedDatasetBuilder

short_musan_dir = '/mnt/sdb/liruiqi/datasets/musan_noise/musan/short-musan'
sr = 24000
save_dir = '/mnt/sdb/liruiqi/SingingDictation/data/binary/musan_24k'
os.makedirs(save_dir, exist_ok=True)
valid_ratio = 0.1
test_ratio = 0.1

# %%
wav_paths = []
for root, dirs, files in os.walk(short_musan_dir):
    if len(files) > 0:
        for f_name in files:
            if Path(f_name).suffix == '.wav':
                wav_paths.append(os.path.join(root, f_name))

musan_noise_dict = {}
for wav_path in tqdm(wav_paths, total=len(wav_paths)):
    item_name = Path(wav_path).stem
    if 'babble' in wav_path:
        _, suf = item_name.split('-')
        if 'train' in item_name:
            item_name = 'babble' + '-' + suf
        elif 'test' in item_name:
            item_name = 'babble' + '-' + str(int(suf) + 8000)
        elif 'valid' in item_name:
            item_name = 'babble' + '-' + str(int(suf) + 9000)
    if item_name in musan_noise_dict:
        print(f'skip {wav_path}: duplicated item name')
        continue
    wav, _ = librosa.core.load(wav_path, sr=sr)
    musan_noise_dict[item_name] = wav.astype(np.float16)

item_names = list(musan_noise_dict.keys())
np.random.shuffle(item_names)

valid_ids = item_names[:int(len(item_names) * valid_ratio)]
test_ids = item_names[int(len(item_names) * valid_ratio): int(len(item_names) * valid_ratio) + int(len(item_names) * test_ratio)]
train_ids = item_names[int(len(item_names) * valid_ratio) + int(len(item_names) * test_ratio): ]

train_set = {k: musan_noise_dict[k] for k in train_ids}
test_set = {k: musan_noise_dict[k] for k in test_ids}
valid_set = {k: musan_noise_dict[k] for k in valid_ids}

def save_feat(feat_dict, prefix, feat_dir):
    builder = IndexedDatasetBuilder(f"{feat_dir}/{prefix}")
    for k in feat_dict:
        feat = feat_dict[k]
        builder.add_item({'item_name': k, 'feat': feat})
    builder.finalize()

# np.save(f"{save_dir}/train.npy", train_set, allow_pickle=True)
# np.save(f"{save_dir}/valid.npy", valid_set, allow_pickle=True)
# np.save(f"{save_dir}/test.npy", test_set, allow_pickle=True)

save_feat(train_set, 'train', save_dir)
save_feat(valid_set, 'valid', save_dir)
save_feat(test_set, 'test', save_dir)
