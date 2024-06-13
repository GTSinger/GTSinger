# dtw code from NeuralSVB
from utils.indexed_datasets import IndexedDataset
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import time
import os
from multiprocessing import Pool
from utils.pitch_utils import f0_to_coarse, denorm_f0
import numpy as np
from utils.hparams import hparams, set_hparams
from utils.indexed_datasets import IndexedDataset

import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit

import torch

@jit
def time_warp(costs):
    dtw = np.zeros_like(costs)
    dtw[0,1:] = np.inf
    dtw[1:,0] = np.inf
    eps = 1e-4
    for i in range(1,costs.shape[0]):
        for j in range(1,costs.shape[1]):
            dtw[i,j] = costs[i,j] + min(dtw[i-1,j],dtw[i,j-1],dtw[i-1,j-1])
    return dtw

def align_from_distances(distance_matrix, debug=False):
    # for each position in spectrum 1, returns best match position in spectrum2
    # using monotonic alignment
    dtw = time_warp(distance_matrix)

    i = distance_matrix.shape[0]-1
    j = distance_matrix.shape[1]-1
    results = [0] * distance_matrix.shape[0]
    while i > 0 and j > 0:
        results[i] = j
        i, j = min([(i-1,j),(i,j-1),(i-1,j-1)], key=lambda x: dtw[x[0],x[1]])

    if debug:
        visual = np.zeros_like(dtw)
        visual[range(len(results)),results] = 1
        plt.matshow(visual)
        plt.show()

    return results

## here is API for one sample
def NaiveDTW(src, tgt, input, mask=None):
    # src: [S, H]
    # tgt: [T, H]
    # mask: [T, 1]
    if mask is not None:
        dists = torch.cdist(src.unsqueeze(0), tgt.unsqueeze(0)) * mask.transpose(-1, -2).unsqueeze(dim=1) # [1, S, T]
    else:
        dists = torch.cdist(src.unsqueeze(0), tgt.unsqueeze(0))  # [1, S, T]
    costs = dists.squeeze(0)  # [S, T]
    alignment = align_from_distances(costs.T.cpu().detach().numpy())
    output = input[alignment]
    return output, alignment

## here is API for one sample  (Zero meaned )
def ZMNaiveDTW(src, tgt, input):
    # src: [S, H]
    # tgt: [T, H]
    src = (src - src.mean())#/src.std()
    tgt = (tgt - tgt.mean())#/tgt.std()
    dists = torch.cdist(src.unsqueeze(0)[:, :, None], tgt.unsqueeze(0)[:, :, None])  # [1, S, T]
    costs = dists.squeeze(0)  # [S, T]
    alignment = align_from_distances(costs.T.cpu().detach().numpy())
    output = input[alignment]
    return output, alignment

## here is API for one sample  (Normalized )
def NNaiveDTW(src, tgt, input):
    # src: [S, H]
    # tgt: [T, H]
    src = (src - src.mean()) / (src.std() + 0.00000001)
    tgt = (tgt - tgt.mean()) / (tgt.std() + 0.00000001)
    dists = torch.cdist(src.unsqueeze(0)[:, :, None], tgt.unsqueeze(0)[:, :, None])  # [1, S, T]
    costs = dists.squeeze(0)  # [S, T]
    alignment = align_from_distances(costs.T.cpu().detach().numpy())
    output = input[alignment]
    return output, alignment

def word_segment(f0, note, mel2word, note2word):
    max_word = max(mel2word)
    # print(max_word)
    word_f0s = []
    word_notes = []
    for i in range(0, max_word+1):
        word_f0s.append(f0[mel2word==i])
        word_notes.append(note[note2word==i])
    return word_f0s, word_notes

if __name__ == '__main__':
    # code for visualization
    def spec_to_figure(spec, vmin=None, vmax=None, name=''):
        if isinstance(spec, torch.Tensor):
            spec = spec.cpu().numpy()
        fig = plt.figure(figsize=(12, 6))
        plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
        plt.savefig(os.path.join('tmp', name))
        return fig

    def f0_to_figure(f0_src, f0_aligned=None, f0_prof=None, name='f0.png'):
        fig = plt.figure(figsize=(12, 8))
        f0_src = f0_src.cpu().numpy()
        # f0_src[f0_src == 0] = np.nan
        plt.plot(f0_src, color='r', label='src')
        if f0_aligned is not None:
            f0_aligned = f0_aligned.cpu().numpy()
            # f0_aligned[f0_aligned == 0] = np.nan
            plt.plot(f0_aligned, color='b', label='f0_aligned')
        if f0_prof is not None:
            f0_pred = f0_prof.cpu().numpy()
            # f0_prof[f0_prof == 0] = np.nan
            plt.plot(f0_pred, color='green', label='profession')
        plt.legend()
        plt.savefig(name)
        return fig

    # set_hparams()

    # train_ds = FastSingingDataset('test')

    # # Test One sample case
    # sample = train_ds[0]
    # amateur_f0 = sample['f0']
    # prof_f0 = sample['prof_f0']

    # amateur_uv = sample['uv']
    # amateur_padding = sample['mel2ph'] == 0
    # prof_uv = sample['prof_uv']
    # prof_padding = sample['prof_mel2ph'] == 0
    # amateur_f0_denorm = denorm_f0(amateur_f0, amateur_uv, hparams, pitch_padding=amateur_padding)
    # prof_f0_denorm = denorm_f0(prof_f0, prof_uv, hparams, pitch_padding=prof_padding)

    # # 用normed_interpolated_f0 如下, 效果更差，下降20个acc..
    # # amateur_f0_denorm = amateur_f0 #denorm_f0(amateur_f0, amateur_uv, hparams, pitch_padding=amateur_padding)
    # # prof_f0_denorm = prof_f0 #denorm_f0(prof_f0, prof_uv, hparams, pitch_padding=prof_padding)

    # amateur_mel = sample['mel']
    # prof_mel = sample['prof_mel']
    # pad_num = max(prof_mel.shape[0] - amateur_mel.shape[0], 0)
    # amateur_mel_padded = F.pad(amateur_mel, [0, 0, 0, pad_num])[:prof_mel.shape[0], :]
    # # aligned_mel, alignment = NaiveDTW(amateur_f0_denorm, prof_f0_denorm, amateur_mel)
    # # aligned_f0_denorm, alignment = NaiveDTW(amateur_f0_denorm, prof_f0_denorm, amateur_f0_denorm)
    # aligned_mel, alignment = ZMNaiveDTW(amateur_f0_denorm, prof_f0_denorm, amateur_mel)
    # aligned_f0_denorm, alignment = ZMNaiveDTW(amateur_f0_denorm, prof_f0_denorm, amateur_f0_denorm)
    # cat_spec = torch.cat([amateur_mel_padded, aligned_mel, prof_mel], dim=-1)
    # spec_to_figure(cat_spec, name=f'f0_denorm_mel_Cn.png')
    # # f0 align f0
    # f0_to_figure(f0_src=amateur_f0_denorm, f0_aligned=aligned_f0_denorm, f0_prof=prof_f0_denorm,
    #              name=f'f0_denorm_f0_Cn.png')
    # amateur_mel2ph = sample['mel2ph']
    # prof_mel2ph = sample['prof_mel2ph']
    # aligned_mel2ph = amateur_mel2ph[alignment]
    # acc = (prof_mel2ph == aligned_mel2ph).sum().cpu().numpy() / (
    #         prof_mel2ph != 0).sum().cpu().numpy()
    # print(acc)
    # exit()
    data_dir = "/home/renyi/hjz/NeuralSeq/data/binary/staff1031"
    indexed_ds = IndexedDataset(f'{data_dir}/test')
    sample = indexed_ds[130]
    print(sample["item_name"])
    f0 = sample["f0"]
    f0 = torch.FloatTensor(f0)
    uv = f0 == 0
    note = sample["note"]
    note2word = sample["note2words"] + 1
    mel2word = sample["mel2word"]
    word_f0s, word_uvs, word_notes = word_segment(f0, uv, note, mel2word, note2word)
    align_notes = []
    for word_f0, word_note in zip(word_f0s, word_notes):
        note_hz = 2 ** ((word_note-69)/12) * 440 * (word_note > 0).astype(np.float)
        word_f0 = torch.FloatTensor(word_f0).unsqueeze(dim=-1)
    # note_hz = F.pad(torch.FloatTensor(note_hz), [0, f0.shape[0] - note_hz.shape[0]]).unsqueeze(dim=-1)
        note_hz = torch.FloatTensor(note_hz).unsqueeze(dim=-1).repeat(1, 10).view(-1, 1)
        aligned_note, alignment = NaiveDTW(note_hz, word_f0, note_hz, mask=(word_f0>0).float())
        # print(alignment)
        # print(aligned_note)
        # print(c)
        align_notes.append(aligned_note)
    align_notes = torch.cat(align_notes, dim=0)
    f0_to_figure(f0_src=align_notes, f0_aligned=align_notes, f0_prof=f0,
                 name=f'f0_denorm_f0_Cn.png')

    note_hz = 2 ** ((note-69)/12) * 440
    print(note_hz)
    # note_hz = torch.FloatTensor(note_hz).unsqueeze(dim=-1)
    # aligned_note, alignment = NaiveDTW(note_hz, f0.unsqueeze(dim=-1), note_hz)
    # f0_to_figure(f0_src=aligned_note, f0_aligned=align_notes, f0_prof=f0,
    #              name=f'f0_denorm_f0_Cn1.png')
    # for idx in range(len(indexed_ds)):
    #     if indexed_ds[idx]["item_name"] == "因为爱情#一次就好#0":
    #         print(idx)
    #         break


# python modules/voice_conversion/dtw/naive_dtw.py --config egs/datasets/audio/PopBuTFy/svc_ppg.yaml

# 因为爱情#一次就好#0 可以证明必须使用word level的 dtw, idx=20
# idx =76 证明需要repeat 10