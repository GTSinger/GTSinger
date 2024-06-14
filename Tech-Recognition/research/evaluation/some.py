# %%
import json
import os.path
from pathlib import Path
from tqdm import tqdm
import pretty_midi
import mir_eval
import numpy as np
import sys
sys.path.append('/mnt/sdb/liruiqi/SingingDictation')
from utils.audio.pitch_utils import midi_onset_eval, midi_offset_eval, midi_pitch_eval
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.audio.pitch_utils import midi_to_hz

# NOTE: need to specify
mid_pred_dir = '/mnt/sdb/liruiqi/SOME/results/m4singer'

m4singer_dir = '/mnt/sdb/liruiqi/datasets/m4singer'
meta_path = f'{m4singer_dir}/meta.json'
items_list = json.load(open(meta_path))
items = {}
item_names = []
for r in tqdm(items_list, desc='Loading meta data.'):
    item_name = r['item_name']
    singer, song_name, sent_id = item_name.split("#")
    r['wav_fn'] = f'{m4singer_dir}/{singer}#{song_name}/{sent_id}.wav'
    items[item_name] = r
    item_names.append(item_name)

# %%
# 把要算的 pred_path 预先算出来
args = []
for item_name in item_names:
    item = items[item_name]
    wav_fn = item['wav_fn']
    mid_gt_path = Path(wav_fn).with_suffix('.mid')
    mid_pred_path = os.path.join(mid_pred_dir, os.path.relpath(Path(mid_gt_path), m4singer_dir))
    args.append([item_name, str(mid_gt_path), str(mid_pred_path)])

def evaluate(item_name, mid_gt_path, mid_pred_path):
    mid_true = pretty_midi.PrettyMIDI(mid_gt_path)
    mid_pred = pretty_midi.PrettyMIDI(mid_pred_path)

    onset_scores = midi_onset_eval(mid_true, mid_pred)
    offset_scores = midi_offset_eval(mid_true, mid_pred)
    pitch_scores = midi_pitch_eval(mid_true, mid_pred)

    return onset_scores, offset_scores, pitch_scores

# %%
onset_scores = np.zeros(3)
offset_scores = np.zeros(3)
pitch_scores = np.zeros(4)
for item_id, (onset_scores_, offset_scores_, pitch_scores_) in multiprocess_run_tqdm(evaluate, args, desc='evaluating some'):
    onset_scores += np.array(onset_scores_)
    offset_scores += np.array(offset_scores_)
    pitch_scores += np.array(pitch_scores_)

# for arg in args:
#     try:
#         onset_scores_, offset_scores_, pitch_scores_ = evaluate(*arg)
#     except:
#         print(arg)
#         break
#     onset_scores += np.array(onset_scores_)
#     offset_scores += np.array(offset_scores_)
#     pitch_scores += np.array(pitch_scores_)

onset_scores /= len(item_names)
offset_scores /= len(item_names)
pitch_scores /= len(item_names)

print(f'onset | precision: {onset_scores[0]:.3f}')
print(f'onset |    recall: {onset_scores[1]:.3f}')
print(f'onset |        f1: {onset_scores[2]:.3f}')
print(f'offset | precision: {offset_scores[0]:.3f}')
print(f'offset |    recall: {offset_scores[1]:.3f}')
print(f'offset |        f1: {offset_scores[2]:.3f}')
print(f'pitch |    precision: {pitch_scores[0]:.3f}')
print(f'pitch |       recall: {pitch_scores[1]:.3f}')
print(f'pitch |           f1: {pitch_scores[2]:.3f}')
print(f'pitch |overlap_ratio: {pitch_scores[3]:.3f}')
