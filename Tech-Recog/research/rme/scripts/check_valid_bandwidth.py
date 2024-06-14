# %%
import os
import sys
from pathlib import Path
import argparse
import traceback

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.feature
from tqdm import tqdm
import soundfile as sf

valid_bitdepth_format = [
    'DOUBLE',
    'PCM_32',
    'FLOAT',
    'ALAC_32',
    'ALAC_20',
    'PCM_24',
    'DWVW_24',
    'ALAC_24',
    'PCM_16',
    'DWVW_16',
    'DPCM_16',
    'ALAC_16'
]

def get_high_freq_ratio(wav, sr, bw_check_start, bw_check_end):
    # wav, _ = librosa.core.load(wav_path, sr=sr)
    n_fft = round(512 * sr / 24000)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    energy = librosa.feature.rms(y=wav, frame_length=n_fft, hop_length=n_fft // 4)

    spec = librosa.stft(y=wav, n_fft=n_fft, hop_length=n_fft // 4)
    mag = np.abs(spec)
    bw_check_start_idx = np.argmin(np.abs(freqs - bw_check_start)).item()
    bw_check_end_idx = np.argmin(np.abs(freqs - bw_check_end)) if bw_check_end < sr // 2 else len(freqs) - 1

    # compute high freq energy
    spec_interest = librosa.util.abs2(mag, dtype=np.float32)
    spec_interest[..., 0, :] *= 0.5
    if n_fft % 2 == 0:
        spec_interest[..., -1, :] *= 0.5
    high_freq_avg_energy = 2 * np.sum(spec_interest[bw_check_start_idx: bw_check_end_idx + 1], axis=-2, keepdims=True) \
        / (bw_check_end_idx - bw_check_start_idx + 1) ** 2
    high_freq_avg_energy = np.sqrt(high_freq_avg_energy)

    ratio = np.mean(high_freq_avg_energy) / np.mean(energy)
    return ratio

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav-dir', type=str, required=True)
    parser.add_argument('--sr', type=int, required=False, default=48000)
    parser.add_argument('--start-freq', type=int, required=False, default=20000)
    parser.add_argument('--end-freq', type=int, required=False, default=24000)
    parser.add_argument('--thr', type=float, required=False, default=-55)
    parser.add_argument('--not-strict', action='store_true', default=False)

    args = parser.parse_args()

    # or just modify this
    # wav_dir = '/mnt/sdb/liruiqi/SingingDictation/data/raw/temp/男声小批量样例1206/男声小批量样例1206'
    # wav_dir = '/mnt/sdb/liruiqi/SingingDictation/data/raw/temp/女声小批量样例1205'
    wav_dir = args.wav_dir
    default_sr = args.sr
    bw_check_start = args.start_freq
    bw_check_end = args.end_freq
    threshold = args.thr

    print(f'Processing under directory {wav_dir}...')
    wav_paths = []
    for root, dirs, files in os.walk(wav_dir):
        if len(files) > 0:
            for f_name in files:
                if Path(f_name).suffix in ['.mp3', '.wav']:
                    wav_paths.append(os.path.join(root, f_name))
    wav_paths = sorted(wav_paths)

    ratios = []
    log_ratios = []
    potential_fault = []
    for wav_path in tqdm(wav_paths, total=len(wav_paths)):
        try:
            wav, sr = librosa.core.load(wav_path, sr=None)
            assert sr == default_sr, f'wav file {wav_path} has sample rate: {sr} other than {default_sr}'
            ratio = get_high_freq_ratio(
                wav, sr=sr, bw_check_start=bw_check_start, bw_check_end=bw_check_end)
            log_ratio = 20 * np.log10(ratio)
            ratios.append(ratio)
            log_ratios.append(log_ratio)
            if log_ratio < threshold:
                potential_fault.append(wav_path)
            # check bitdepth
            bd = sf.SoundFile(wav_path).subtype
            assert bd in valid_bitdepth_format, f'wav file {wav_path} has wrong bitdepth: {bd}'
        except Exception as err:
            if args.not_strict:
                print(f'Skip {wav_path} for exception {err}')
                continue
            else:
                traceback.print_exc()
                sys.exit(-1)
    avg_ratio = np.mean(ratios)
    avg_log_ratio = 20 * np.log10(avg_ratio)

    print(avg_log_ratio)

    if avg_log_ratio < -50 or len(potential_fault) / len(wav_paths) > 0.8:
        print(f'This batch potentially has effect bandwidth lower than {default_sr // 2}')
    elif len(potential_fault) / len(wav_paths) > 0.1:
        print(f'The whole batch is good, however,')
        print(f'[Warning] These samples (about {len(potential_fault) / len(wav_paths) * 100:.3f}%) potentially have effect bandwidth lower than {default_sr // 2}:')
        print('\n'.join(potential_fault))
        print('Note that if the whole batch is recorded in the same equipment setting, the above warnings can just be false positives.')
    else:
        print('Everything good.')

sys.exit()

# python research/rme/scripts/check_valid_bandwidth.py \
#     --wav-dir /mnt/sdb/liruiqi/SingingDictation/data/raw/temp/男声第1批231220

# %%
# wav_dir = '/mnt/sdb/liruiqi/SingingDictation/data/raw/temp/男声小批量样例1206/男声小批量样例1206'
# wav_dir = '/mnt/sdb/liruiqi/SingingDictation/data/raw/temp/女声小批量样例1205'
wav_dir = '/mnt/sdb/liruiqi/SingingDictation/data/raw/temp/【12.17】实验室女声'
default_sr = 48000
bw_check_start = 20000
bw_check_end = 24000
threshold = -50

wav_paths = []
for root, dirs, files in os.walk(wav_dir):
    if len(files) > 0:
        for f_name in files:
            if Path(f_name).suffix in ['.mp3', '.wav']:
                wav_paths.append(os.path.join(root, f_name))
wav_paths = sorted(wav_paths)

ratios = []
log_ratios = []
potential_fault = []
for wav_path in tqdm(wav_paths, total=len(wav_paths)):
    wav, sr = librosa.core.load(wav_path, sr=default_sr)
    ratio = get_high_freq_ratio(
        wav, sr=sr, bw_check_start=bw_check_start, bw_check_end=bw_check_end)
    log_ratio = 20 * np.log10(ratio)
    ratios.append(ratio)
    log_ratios.append(log_ratio)
    if log_ratio < threshold:
        potential_fault.append(wav_path)
    # check bitdepth
    bd = sf.SoundFile(wav_path).subtype
    assert bd in valid_bitdepth_format, f'wav file {wav_path} has wrong bitdepth: {bd}'
avg_ratio = np.mean(ratios)
avg_log_ratio = 20 * np.log10(avg_ratio)

print(avg_ratio)
print(avg_log_ratio)

# %%
# wav_path = '/mnt/sdb/liruiqi/SingingDictation/data/raw/temp/男声小批量样例1206/男声小批量样例1206/强弱/爱情转移/男声_无技巧_句1.wav'
# wav_path = '/mnt/sdb/liruiqi/SingingDictation/data/raw/temp/女声小批量样例1205/女声 12-5_/假声 升1key（一整段）/女声002-升一key.wav'
# wav_path = '/mnt/sdb/liruiqi/SingingDictation/data/raw/temp/女声第1批/爱情转移/假声/女声_弱假声.wav'
# wav_path = '/mnt/sdb/liruiqi/SingingDictation/data/raw/temp/女声第1批/单身情歌/假声/女声_弱假声.wav'
# wav_path = '/mnt/sdb/liruiqi/SingingDictation/data/raw/temp/女声-无技巧-句7.wav'
wav_path = '/mnt/sdb/liruiqi/SingingDictation/data/raw/temp/女声-假声-句1.wav'
sr = 48000
wav, _ = librosa.core.load(wav_path, sr=sr)
n_fft = round(512 * sr / 24000)
freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

# energy = librosa.feature.rms(y=wav, frame_length=n_fft, hop_length=n_fft // 4)


spec = librosa.stft(y=wav, n_fft=n_fft, hop_length=n_fft // 4)
mag = np.abs(spec)
mag_squared = librosa.util.abs2(mag, dtype=np.float32)
energy_sum_squared = 2 * np.sum(mag_squared, axis=-2)

bw_check_start_idx = np.argmin(np.abs(freqs - bw_check_start)).item()
bw_check_end_idx = np.argmin(np.abs(freqs - bw_check_end)) if bw_check_end < sr // 2 else len(freqs) - 1

# compute high freq energy
high_freq_ess = 2 * np.sum(mag_squared[bw_check_start_idx: bw_check_end_idx + 1], axis=-2) \
    / (bw_check_end_idx - bw_check_start_idx + 1) ** 2

ratio = (high_freq_ess / energy_sum_squared).squeeze()

sorted_ratio = sorted(ratio, reverse=True)
sorted_log_ratio = 20 * np.log10(sorted_ratio)


