import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import pyloudnorm as pyln
from skimage.transform import resize
import struct
import webrtcvad
from scipy.ndimage import binary_dilation
import argparse
import torch
import data_process.utils.plot as plot

def process_wav(hparams,wav_path):

    x, sr = librosa.load(wav_path, sr=48000)

    # compute power spectrogram with stft(short-time fourier transform):
    # 基于stft，计算power spectrogram
    spectrogram = librosa.amplitude_to_db(librosa.stft(x))

    # show
    librosa.display.specshow(spectrogram, y_axis='linear',sr=48000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('power spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()
    plt.savefig('test.png')


def plot_mel( spec_out, spec_gt=None, name=None, title='', f0s=None, dur_info=None):
    vmin = hparams['mel_vmin']
    vmax = hparams['mel_vmax']
    if len(spec_out.shape) == 3:
        spec_out = spec_out[0]
    if spec_gt is not None:
        if len(spec_gt.shape) == 3:
            spec_gt = spec_gt[0]
        max_len = max(len(spec_gt), len(spec_out))
        if max_len - len(spec_gt) > 0:
            spec_gt = np.pad(spec_gt, [[0, max_len - len(spec_gt)], [0, 0]], mode='constant',
                                constant_values=vmin)
        if max_len - len(spec_out) > 0:
            spec_out = np.pad(spec_out, [[0, max_len - len(spec_out)], [0, 0]], mode='constant',
                                constant_values=vmin)
        spec_out = np.concatenate([spec_out, spec_gt], -1)
    # name = f'mel_val_{batch_idx}' if name is None else name
    spec_to_figure(
        spec_out, vmin, vmax, title=title, f0s=f0s, dur_info=dur_info)

def spec_to_figure(spec, vmin=None, vmax=None, title='', f0s=None, dur_info=None):
    H = spec.shape[1] // 2
    fig = plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    if dur_info is not None:
        assert isinstance(dur_info, dict)
        txt = dur_info['txt']
        dur_gt = dur_info['dur_gt']
        dur_gt = np.cumsum(dur_gt).astype(int)
        for i in range(len(dur_gt)):
            shift = (i % 8) + 1
            plt.text(dur_gt[i], shift * 4, txt[i])
            plt.vlines(dur_gt[i], 0, H // 2, colors='b')  # blue is gt
        plt.xlim(0, dur_gt[-1])
        if 'dur_pred' in dur_info:
            dur_pred = dur_info['dur_pred']
            if isinstance(dur_pred, torch.Tensor):
                dur_pred = dur_pred.cpu().numpy()
            dur_pred = np.cumsum(dur_pred).astype(int)
            for i in range(len(dur_pred)):
                shift = (i % 8) + 1
                plt.text(dur_pred[i], H + shift * 4, txt[i])
                plt.vlines(dur_pred[i], H, H * 1.5, colors='r')  # red is pred
            plt.xlim(0, max(dur_gt[-1], dur_pred[-1]))
    if f0s is not None:
        ax = plt.gca()
        ax2 = ax.twinx()
        if not isinstance(f0s, dict):
            f0s = {'f0': f0s}
        for i, (k, f0) in enumerate(f0s.items()):
            ax2.plot(f0, label=k, linewidth=1, alpha=0.5)
        ax2.set_ylim(0, 1000)
        ax2.legend()
    plt.savefig('./test.png')
    # return fig

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
        "loud_norm": False,
        "mel_vmin": -6,
        "mel_vmax": 1.5
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav-path', type=str, required=True)
    args = parser.parse_args()
    
    wav_path = args.wav_path
    
    f01=process_wav(hparams,wav_path=wav_path)

