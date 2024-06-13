import numpy as np
# import pyworld as pw
import json
import matplotlib.pyplot as plt
import librosa
import numpy as np
import pyloudnorm as pyln
from skimage.transform import resize
import struct
import webrtcvad
from scipy.ndimage import binary_dilation
import argparse
import data_process.utils.plot as plot


def process_wav(hparams,wav_path):

    wav2spec_dict = plot.librosa_wav2spec(
        wav_path,
        fft_size=hparams['fft_size'],
        hop_size=hparams['hop_size'],
        win_length=hparams['win_size'],
        num_mels=hparams['audio_num_mel_bins'],
        fmin=hparams['fmin'],
        fmax=hparams['fmax'],
        sample_rate=hparams['audio_sample_rate'],
        loud_norm=hparams['loud_norm'])
    mel = wav2spec_dict['mel']
    wav = wav2spec_dict['wav'].astype(np.float16)

    # # print(sr,osr)
    # if osr != sr:
    #     y = librosa.resample(y, osr, sr)

    #parselmouth
    time_step = hparams['hop_size'] / hparams['audio_sample_rate'] * 1000
    f0_min = 80
    f0_max = 800

    if hparams['hop_size'] == 128:
        pad_size = 4
    elif hparams['hop_size'] == 256:
        pad_size = 2
    else:
        assert False
    import parselmouth
    f0 = parselmouth.Sound(wav, hparams['audio_sample_rate']).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    lpad = pad_size * 2
    rpad = len(mel) - len(f0) - lpad
    f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
    delta_l = len(mel) - len(f0)
    assert np.abs(delta_l) <= 8
    if delta_l > 0:
        f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
    # print(len(mel),len(f0))
    f0 = f0[:len(mel)]

    return f0

def plot_f0(f01,f02):
    plt.figure()
    plt.plot(f01,label='f0_1')
    plt.plot(f02,label='f0_2')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('test.png')


if __name__ == '__main__':
    hparams={
        "audio_sample_rate": 48000,
        "hop_size": 256,
        "win_size": 1024,
        "fft_size": 1024,
        "fmax": 24000,
        "fmin": 20,
        "max_frames": 3000,
        "audio_num_mel_bins":80,
        "loud_norm": False
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav1', type=str, required=True)
    parser.add_argument('--wav2',type=str,required=True)
    args = parser.parse_args()
    wav1 = args.wav1
    wav2 = args.wav2
    f01=process_wav(hparams,wav1)
    f02=process_wav(hparams,wav2)
    plot_f0(f01,f02)
