# %%
import os
import glob
from pathlib import Path
import math

from tqdm import tqdm
import torch
import mir_eval
import librosa
import numpy as np

from modules.pe.rmvpe import RMVPE
from utils.audio import get_wav_num_frames
from utils.commons.dataset_utils import batch_by_size

pe_ckpt = 'checkpoints/rmvpe/model.pt'
rmvpe = None
sr = 48000
hop_size = 256

"""
这个是用来计算gt的，即gt vocoder和gt wav之间的距离。目前仅限于 rms
"""


# wav_dir = '/mnt/sdb/liruiqi/RMSSinger_dev/checkpoints/svs/240118-dpost-rms-01/generated_120000_/wavs'
# wav_dir = '/mnt/sdb/liruiqi/RMSSinger_dev/checkpoints/svs/diff_dpost/generated_160000_/wavs'
wav_dir = '/mnt/sdb/liruiqi/RMSSinger_dev/checkpoints/svs/240112-dpost-rms-01/generated_125000_/wavs'

# %%
item_names = []
items = {}
for wav_path in sorted(glob.glob(f"{wav_dir}/*.wav")):
    item_name = Path(wav_path).stem
    item_names.append(item_name)
    if '[P]' in wav_path:
        items[item_name] = {'wav_path': wav_path.replace('[P]', '[G]')}
    else:
        spk, song_name, idx = item_name.split('#')
        idx = idx[:-3]
        gt_path = f"/mnt/sdb/liruiqi/datasets/230227seg/{spk}/{song_name}/{song_name}#{idx}.wav"
        items[item_name] = {'wav_path': gt_path}

# %%
if rmvpe is None:
    rmvpe = RMVPE(pe_ckpt, device='cuda')
id_and_sizes = []
for idx, item_name in enumerate(item_names):
    item = items[item_name]
    wav_path = item['wav_path']
    total_frames = get_wav_num_frames(wav_path, sr)
    id_and_sizes.append((idx, round(total_frames / hop_size)))
get_size = lambda x: x[1]
bsz = 128
max_tokens = 240000
bs = batch_by_size(id_and_sizes, get_size, max_tokens=max_tokens, max_sentences=32)
for i in range(len(bs)):
    bs[i] = [bs[i][j][0] for j in range(len(bs[i]))]

for batch in tqdm(bs, total=len(bs), desc=f'| Processing f0 in [max_tokens={max_tokens}; max_sentences={bsz}]'):
    wavs, mel_lengths, lengths = [], [], []
    for idx in batch:
        item_name = item_names[idx]
        item = items[item_name]
        wav_fn = item['wav_path']
        wav, _ = librosa.core.load(wav_fn, sr=sr)
        wavs.append(wav)
        mel_lengths.append(math.ceil((wav.shape[0] + 1) / hop_size))
        # lengths.append((wav.shape[0] + rmvpe.mel_extractor.hop_length - 1) // rmvpe.mel_extractor.hop_length)
        lengths.append((wav.shape[0] + hop_size - 1) // hop_size)

    with torch.no_grad():
        f0s, uvs = rmvpe.get_pitch_batch(
            wavs, sample_rate=sr,
            hop_size=hop_size,
            lengths=lengths,
            fmax=900,
            fmin=50
        )
    torch.cuda.empty_cache()

    for i in range(len(f0s)):
        items[item_names[batch[i]]]['f0'] = f0s[i]

if rmvpe is not None:
    rmvpe.release_cuda()
    torch.cuda.empty_cache()

# %%
vrs, vfas, rpas, rcas, oas = [], [], [], [], []
for item_name in item_names:
    if '[G]' in item_name:
        gt_item = items[item_name]
        pred_item = items[item_name.replace('[G]', '[P]')]
    else:
        continue
    f0_gt = freq_gt = gt_item['f0']
    f0_pred = freq_pred = pred_item['f0']
    step = hop_size / sr
    t_gt = np.arange(0, step * len(f0_gt), step)
    t_pred = np.arange(0, step * len(f0_pred), step)

    if len(t_gt) > len(freq_gt):
        t_gt = t_gt[:len(freq_gt)]
    else:
        freq_gt = freq_gt[:len(t_gt)]
    if len(t_pred) > len(freq_pred):
        t_pred = t_pred[:len(freq_pred)]
    else:
        freq_pred = freq_pred[:len(t_pred)]

    ref_voicing, ref_cent, est_voicing, est_cent = mir_eval.melody.to_cent_voicing(t_gt, freq_gt,
                                                                                   t_pred, freq_pred)
    vr, vfa = mir_eval.melody.voicing_measures(ref_voicing,
                                               est_voicing)  # voicing recall, voicing false alarm
    rpa = mir_eval.melody.raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
    rca = mir_eval.melody.raw_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
    oa = mir_eval.melody.overall_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)

    vrs.append(vr)
    vfas.append(vfa)
    rpas.append(rpa)
    rcas.append(rca)
    oas.append(oa)

print(f'Results for {int(len(items)/2)} pairs:')
print(f'melody |          VR: {np.mean(vrs):.3f}')
print(f'melody |         VFA: {np.mean(vfas):.3f}')
print(f'melody |*        RPA: {np.mean(rpas):.3f}')
print(f'melody |         RCA: {np.mean(rcas):.3f}')
print(f'melody |*         OA: {np.mean(oas):.3f}')

