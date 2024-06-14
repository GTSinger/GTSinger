import math

from tqdm import tqdm
import librosa
import torch

from modules.pe.rmvpe import RMVPE
from utils.commons.dataset_utils import batch_by_size
from utils.audio import get_wav_num_frames

"""
一个方便直接调用且 batch inference 的接口
"""

rmvpe_ckpt = ''
rmvpe = None


def extract(wav_fns: list, id_and_sizes=None, ckpt=None, sr=24000, hop_size=128, bsz=128, max_tokens=100000,
             fmax=900, fmin=50, device='cuda'):
    assert ckpt is not None
    global rmvpe
    if rmvpe is None:
        rmvpe = RMVPE(ckpt, device=device)
    if id_and_sizes is None:
        id_and_sizes = []
        if type(wav_fns[0]) == str:    # wav_paths
            for idx, wav_path in enumerate(wav_fns):
                total_frames = get_wav_num_frames(wav_path, sr)
                id_and_sizes.append((idx, round(total_frames / hop_size)))
        else:                       # numpy arrays, mono wavs
            for idx, wav in enumerate(wav_fns):
                id_and_sizes.append((idx, round(wav.shape[-1] / hop_size)))
    get_size = lambda x: x[1]
    bs = batch_by_size(id_and_sizes, get_size, max_tokens=max_tokens, max_sentences=bsz)
    for i in range(len(bs)):
        bs[i] = [bs[i][j][0] for j in range(len(bs[i]))]

    f0_res = [None] * len(wav_fns)
    for batch in tqdm(bs, total=len(bs), desc=f'| Processing f0 in [max_tokens={max_tokens}; max_sentences={bsz}]'):
        wavs, mel_lengths, lengths = [], [], []
        for idx in batch:
            if type(wav_fns[idx]) == str:
                wav_fn = wav_fns[idx]
                wav, _ = librosa.core.load(wav_fn, sr=sr)
            else:
                wav = wav_fns[idx]
            wavs.append(wav)
            mel_lengths.append(math.ceil((wav.shape[0] + 1) / hop_size))
            lengths.append((wav.shape[0] + hop_size - 1) // hop_size)

        with torch.no_grad():
            f0s, uvs = rmvpe.get_pitch_batch(
                wavs, sample_rate=sr,
                hop_size=hop_size,
                lengths=lengths,
                fmax=fmax,
                fmin=fmin
            )

        for i, idx in enumerate(batch):
            f0_res[idx] = f0s[i]

    if rmvpe is not None:
        rmvpe.release_cuda()
        torch.cuda.empty_cache()
        rmvpe = None

    return f0_res
