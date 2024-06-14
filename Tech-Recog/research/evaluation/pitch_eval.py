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

m4_spks = ['Alto-4', 'Bass-2', 'Tenor-7', 'Soprano-3', 'Alto-1', 'Tenor-5', 'Alto-3', 'Tenor-3', 'Tenor-6', 'Alto-5',
            'Tenor-4', 'Alto-7', 'Alto-6', 'Bass-1', 'Soprano-1', 'Tenor-2', 'Tenor-1', 'Alto-2', 'Bass-3',
            'Soprano-2']
rms_spks = ["tomake", "一生守候", "世纪之光", "因为爱情", "在夜里跳舞", "夜夜夜夜", "少女音", "我喜欢", "永不失联的爱",
            "没离开过", "流浪记", "爱要坦荡荡", ]
xiaoma_spks = ['_张怡爽', '_张sir', '男_1_1-', '_傅琳小玥', '_于旺博', '_拉孜娜', '_常宇菲', '男_1_3-', '_关皓聪', '_张依依',
               '_小胡', '_楚轩明', '男_2_5-', '_james', '_小张', '女_2_1-', '女_1_1-', '_沈萌萌', '_赵飞', '_Cc', '_赵友豪',
               '_丁文凯', '_茉莉', '_pucc', '_孙若琳', '_兰庭', '_张田杨', '_张倬菡', '_samp', '_zhong', '_吕洋', '_贾旎',
               '_采萱', '_罗莹璐', '男_2_1-', '男_1_4-', '_朱苡葳', '女_2_6-', '男_1_2-', '_全方怡', '_阿单', '男_2_4-',
               '女_2_5-', '_成佳怡', '_胡某人', '_小余', '_蔡丽申', '_刘雅晴', '_依帕尔', '_陈思雨', '男_2_3-', '_YYH',
               '_陶子', '_李湘毅', '_王成豪', '_小陈', '_油桃', '_二十四', '_陈宁', '_月林', '女_1_2-', '_孙祥', '_阿廷',
               '_张若涵', '_王佳妮', '_陈仙旦', '_李嘉英', '男_2_2-', '_花花', '_小武', '_王琳琳', '_陈庆之', '_候青德',
               '_王梦汐', '_不知火舞的姐姐', '_刘晋佟', '_刘畅', '_郭宁宁', '_王毅思', '女_2_3-', '_骨头老姐', '_葛里',
               '_李品一', '_小胖丁', '_李佩彤', '女_2_2-', '_杨沁颖', '_汤紫棋', '女_2_4-', '_李梦甜', '_李美泽', '_濛',
               '_赵禧桡']
sts_spks = ['男2号', '男1号', '女10号', '张蔷-女', '女6号', '女5号', '庞胜楠-女', '男4号', '女12号', '关苏庭-女',
            '男3号', '女2号', '女11号', '女9号', '女1号', '李俊霖-女', '男5号', '女4号', '女13号', '女8号']
en_spks = ['en']

# 不同的比例
# wav_dir = '/mnt/sdb/liruiqi/RMSSinger_dev/checkpoints/svs/240112-dpost-rms-01/generated_125000_/wavs'
# wav_dir = '/mnt/sdb/liruiqi/RMSSinger_dev/checkpoints/svs/240118-dpost-rms-01/generated_120000_/wavs'
# wav_dir = '/mnt/sdb/liruiqi/RMSSinger_dev/checkpoints/svs/240118-dpost-rms-03/generated_160000_/wavs'
# wav_dir = '/mnt/sdb/liruiqi/RMSSinger_dev/checkpoints/svs/240118-dpost-rms-04/generated_135000_/wavs'
# wav_dir = '/mnt/sdb/liruiqi/RMSSinger_dev/checkpoints/svs/240118-dpost-rms-05/generated_160000_/wavs'
# wav_dir = '/mnt/sdb/liruiqi/RMSSinger_dev/checkpoints/svs/240118-dpost-rms-06/generated_160000_/wavs'
# 最初的，好像是用xiaoma训的
# wav_dir = '/mnt/sdb/liruiqi/RMSSinger_dev/checkpoints/svs/diff_dpost/generated_160000_/wavs'
# m4+rms
# wav_dir = '/mnt/sdb/liruiqi/RMSSinger_dev/checkpoints/svs/240115-dpost-rms-01/generated_115000_/wavs'
# large
# wav_dir = '/mnt/sdb/liruiqi/RMSSinger_dev/checkpoints/svs/240119-dpost-large-01/generated_255000_/wavs'
wav_dir = '/mnt/sdb/liruiqi/RMSSinger_dev/checkpoints/svs/240119-dpost-large-02/generated_200000_/wavs'


# %%
item_names = []
items = {}
for wav_path in sorted(glob.glob(f"{wav_dir}/*.wav")):
    item_name = Path(wav_path).stem
    spk = item_name.split('#')[0]
    if spk in sts_spks:
        continue
    # if spk in en_spks:
    #     continue
    if spk in xiaoma_spks:
        continue
    if spk in rms_spks:
        continue
    if spk in m4_spks:
        continue
    item_names.append(item_name)
    items[item_name] = {'wav_path': wav_path}

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
max_tokens = 400000
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
    # torch.cuda.empty_cache()

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

vrs = sorted(vrs, reverse=True)
vfas = sorted(vfas, reverse=True)
rpas = sorted(rpas, reverse=True)
rcas = sorted(rcas, reverse=True)
oas = sorted(oas, reverse=True)

print(f'Results for {int(len(items)/2)} pairs:')
print(f'melody |   VR: {np.mean(vrs):.3f} | {np.mean(vrs[:int(0.8*len(vrs))]):.3f} | {np.mean(vrs[:int(0.5*len(vrs))]):.3f} | {np.mean(vrs[:int(0.3*len(vrs))]):.3f}')
print(f'melody |  VFA: {np.mean(vfas):.3f} | {np.mean(vfas[:int(0.8*len(vfas))]):.3f} | {np.mean(vfas[:int(0.5*len(vfas))]):.3f} | {np.mean(vfas[:int(0.3*len(vfas))]):.3f}')
print(f'melody |* RPA: {np.mean(rpas):.3f} | {np.mean(rpas[:int(0.8*len(rpas))]):.3f} | {np.mean(rpas[:int(0.5*len(rpas))]):.3f} | {np.mean(rpas[:int(0.3*len(rpas))]):.3f}')
print(f'melody |  RCA: {np.mean(rcas):.3f} | {np.mean(rcas[:int(0.8*len(rcas))]):.3f} | {np.mean(rcas[:int(0.5*len(rcas))]):.3f} | {np.mean(rcas[:int(0.3*len(rcas))]):.3f}')
print(f'melody |*  OA: {np.mean(oas):.3f} | {np.mean(oas[:int(0.8*len(oas))]):.3f} | {np.mean(oas[:int(0.5*len(oas))]):.3f} | {np.mean(oas[:int(0.3*len(oas))]):.3f}')

# %%
# row2
# melody |* RPA: 0.622 | 0.681 | 0.739 | 0.774

# row3
# melody |* RPA: 0.525 | 0.570 | 0.618 | 0.657

# row4
# melody |* RPA: 0.564 | 0.610 | 0.678 | 0.724

# row5
# melody |* RPA: 0.613 | 0.659 | 0.701 | 0.734

# row6
# melody |* RPA: 0.623 | 0.660 | 0.698 | 0.726

# row7
# melody |* RPA: 0.547 | 0.595 | 0.637 | 0.676

# row8


# row9


# row10
# melody |* RPA: 0.548 | 0.600 | 0.666 | 0.713
# melody |* RPA: 0.625 | 0.669 | 0.704 | 0.728

# row11
# m4         melody |* RPA: 0.550 | 0.600 | 0.674 | 0.734
# rms        melody |* RPA: 0.556 | 0.595 | 0.650 | 0.695
# xiaoma     melody |* RPA: 0.469 | 0.492 | 0.529 | 0.555
# rms+xiaoma melody |* RPA: 0.533 | 0.568 | 0.621 | 0.672
