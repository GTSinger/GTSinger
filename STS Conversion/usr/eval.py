import sys
import warnings
warnings.filterwarnings("ignore")

import os
import multiprocessing as mp
import re
import traceback
import librosa
import numpy as np
from tqdm import tqdm
import parselmouth
import mir_eval
import torch
from scipy.ndimage import gaussian_filter

from utils import audio
from vocoders.hifigan import HifiGAN
# from utils.hparams import hparams, set_hparams
from data_gen.tts.data_gen_utils import get_pitch

# %%
class Evaluation:
    def __init__(self):
        # set_hparams()
        self.work_dir = f"checkpoints/{hparams['exp_name']}"
        if hparams['lsd'] == 'lsd1':
            self.lsd = Evaluation.lsd1
        elif hparams['lsd'] == 'lsd2':
            self.lsd = Evaluation.lsd2
        self.get_f_list()
        if hparams['dataset'] == 'speech2singing':
            self.sing_tag, self.speech_tag = '专业', '歌词'

    @staticmethod
    def lsd1(wav1, wav2):
        g_spec = librosa.core.stft(wav1, n_fft=hparams['fft_size'], hop_length=hparams['hop_size'],
                                   win_length=hparams['win_size'])
        p_spec = librosa.core.stft(wav2, n_fft=hparams['fft_size'], hop_length=hparams['hop_size'],
                                   win_length=hparams['win_size'])
        g_spec = g_spec[librosa.fft_frequencies(hparams["audio_sample_rate"], hparams['fft_size']) > 100, :]
        p_spec = p_spec[librosa.fft_frequencies(hparams["audio_sample_rate"], hparams['fft_size']) > 100, :]
        g_spec = np.abs(g_spec) ** 2
        p_spec = np.abs(p_spec) ** 2
        min_len = min(g_spec.shape[1], p_spec.shape[1])
        g_spec, p_spec = g_spec[:, :min_len], p_spec[:, :min_len]
        lsd = np.sqrt(np.mean(np.power(20 * np.log10(g_spec / p_spec), 2)))

        return lsd

    @staticmethod
    def lsd2(wav1, wav2):
        g_spec = librosa.core.stft(wav1, n_fft=hparams['fft_size'], hop_length=hparams['hop_size'],
                                   win_length=hparams['win_size'])
        p_spec = librosa.core.stft(wav2, n_fft=hparams['fft_size'], hop_length=hparams['hop_size'],
                                   win_length=hparams['win_size'])
        g_spec = g_spec[librosa.fft_frequencies(hparams["audio_sample_rate"], hparams['fft_size']) > 100, :]
        p_spec = p_spec[librosa.fft_frequencies(hparams["audio_sample_rate"], hparams['fft_size']) > 100, :]
        g_spec = np.log(0.1 + np.abs(g_spec))
        p_spec = np.log(0.1 + np.abs(p_spec))
        min_len = min(g_spec.shape[1], p_spec.shape[1])
        g_spec, p_spec = g_spec[:, :min_len], p_spec[:, :min_len]
        lsd = np.mean(np.sqrt(np.sum((g_spec - p_spec) ** 2, axis=0)))

        return lsd

    @staticmethod
    def rca(wav1, wav2):
        f01 = Evaluation.get_pitch(wav1)
        f02 = Evaluation.get_pitch(wav2)
        min_len = min(f01.shape[0], f02.shape[0])
        f01, f02 = f01[:min_len], f02[:min_len]
        time_step = hparams['hop_size'] / hparams['audio_sample_rate'] * 1000
        t = np.ones_like(f01) * time_step
        t = np.cumsum(t)
        ref_voicing, ref_cent, est_voicing, est_cent = mir_eval.melody.to_cent_voicing(t, f01, t, f02)
        rca = mir_eval.melody.raw_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
        return rca

    @staticmethod
    def rrd(wav1, wav2):
        _, mel1 = Evaluation.wav2spec(wav1)
        _, mel2 = Evaluation.wav2spec(wav2)

        mel1, mel2 = torch.Tensor(mel1), torch.Tensor(mel2)
        energy1 = torch.Tensor((mel1.exp() ** 2).sum(-1).sqrt())
        rhythm1 = torch.sigmoid((energy1 - energy1.mean())
                                           / (energy1.std() + 1e-8) * hparams.get('sing_spenv_reduced_coef', 1.0))
        rhythm1 = torch.Tensor(gaussian_filter(rhythm1, sigma=hparams.get('spenv_filter', 0)))

        energy2 = torch.Tensor((mel2.exp() ** 2).sum(-1).sqrt())
        rhythm2 = torch.sigmoid((energy2 - energy2.mean())
                                / (energy2.std() + 1e-8) * hparams.get('sing_spenv_reduced_coef', 1.0))
        rhythm2 = torch.Tensor(gaussian_filter(rhythm2, sigma=hparams.get('spenv_filter', 0)))

        dist = torch.dist(rhythm1, rhythm2).item()

        # print(dist)

        return dist

    @staticmethod
    def wav2spec(wav):
        fft_size, hop_size, num_mels, fmin, fmax, sample_rate = hparams['fft_size'], hparams['hop_size'], \
                                                                hparams['audio_num_mel_bins'], hparams['fmin'], \
                                                                hparams['fmax'], hparams['audio_sample_rate']
        # get amplitude spectrogram
        x_stft = librosa.stft(wav, n_fft=hparams['fft_size'], hop_length=hparams['hop_size'],
                              win_length=hparams['win_size'], window='hann', pad_mode="constant")
        spc = np.abs(x_stft)  # (n_bins, T)

        # get mel basis
        fmin = 0 if fmin == -1 else fmin
        fmax = sample_rate / 2 if fmax == -1 else fmax
        mel_basis = librosa.filters.mel(sample_rate, fft_size, num_mels, fmin, fmax)
        mel = mel_basis @ spc

        mel = np.log10(np.maximum(1e-10, mel))  # (n_mel_bins, T)

        l_pad, r_pad = audio.librosa_pad_lr(wav, fft_size, hop_size, 1)
        wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
        wav = wav[:mel.shape[1] * hop_size]

        return wav, mel

    @staticmethod
    def get_pitch(wav):
        time_step = hparams['hop_size'] / hparams['audio_sample_rate'] * 1000
        f0_min = 80
        f0_max = 750
        f0 = parselmouth.Sound(wav, hparams['audio_sample_rate']).to_pitch_ac(
            time_step=time_step / 1000, voicing_threshold=0.6,
            pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
        return f0

    def get_f_list(self):
        self.f_list = {}
        for gen_dir in sorted(os.listdir(self.work_dir)):
            f_list = []
            if gen_dir[:10] != 'generated_':
                continue
            for f_name in sorted(os.listdir(f"{self.work_dir}/{gen_dir}/wavs/")):
                if hparams['voc'] and '[G]' in f_name:
                    f_list.append(f_name)
                elif not hparams['voc'] and '[P]' in f_name:
                    f_list.append(f_name)
            self.f_list[gen_dir] = f_list

    def compute_LSD(self):
        for gen_dir, f_list in sorted(self.f_list.items(), key=lambda x: x[0]):

            results = []
            for i, f_name in tqdm(enumerate(f_list), desc=f'{gen_dir}', ncols=80, leave=False, total=len(f_list)):
                item_name = re.findall(re.compile(r"[\[](.*?)[]]", re.S), f_name)[0]
                spk, song_name, song_idx = item_name.split("#")
                if os.path.exists(f"{hparams['processed_data_dir']}/wav/{spk}#{song_name}#{self.sing_tag}") and \
                        os.path.exists(f"{hparams['processed_data_dir']}/wav/{spk}#{song_name}#{self.sing_tag}/{spk}#{song_name}#{self.sing_tag}#{song_idx}.wav"):
                    g_wav, _ = librosa.core.load(f"{hparams['processed_data_dir']}/wav/{spk}#{song_name}#{self.sing_tag}/{spk}#{song_name}#{self.sing_tag}#{song_idx}.wav",
                                                 sr=hparams.get("audio_sample_rate"))
                else:
                    continue
                p_wav, _ = librosa.core.load(f"{self.work_dir}/{gen_dir}/wavs/{f_name}",
                                             sr=hparams.get("audio_sample_rate"))

                lsd = self.lsd(g_wav, p_wav)
                results.append((f_name[:-4], lsd))

            lsds = np.array([result[1] for result in results])
            lsds.sort()
            avg_lsd = np.mean(lsds[~np.isinf(lsds)])

            if hparams['voc']:
                print(f"LSD: Avg for {self.work_dir} {gen_dir} vocoder: {avg_lsd:.4f}")
            else:
                print(f"LSD: Avg for {self.work_dir} {gen_dir}: {avg_lsd:.4f}")

            if hparams['voc']:
                break

    def compute_RCA(self):
        for gen_dir, f_list in sorted(self.f_list.items(), key=lambda x: x[0]):

            results = []
            for i, f_name in tqdm(enumerate(f_list), desc=f'{gen_dir}', ncols=80, leave=False, total=len(f_list)):
                item_name = re.findall(re.compile(r"[\[](.*?)[]]", re.S), f_name)[0]
                spk, song_name, song_idx = item_name.split("#")
                if os.path.exists(f"{hparams['processed_data_dir']}/wav/{spk}#{song_name}#{self.sing_tag}") and \
                        os.path.exists(f"{hparams['processed_data_dir']}/wav/{spk}#{song_name}#{self.sing_tag}/{spk}#{song_name}#{self.sing_tag}#{song_idx}.wav"):
                    g_wav, _ = librosa.core.load(f"{hparams['processed_data_dir']}/wav/{spk}#{song_name}#{self.sing_tag}/{spk}#{song_name}#{self.sing_tag}#{song_idx}.wav",
                                                 sr=hparams.get("audio_sample_rate"))
                else:
                    continue
                p_wav, _ = librosa.core.load(f"{self.work_dir}/{gen_dir}/wavs/{f_name}",
                                             sr=hparams.get("audio_sample_rate"))

                rca = self.rca(g_wav, p_wav)
                results.append((f_name[:-4], rca))

            rcas = np.array([result[1] for result in results])
            rcas = np.array(sorted(rcas, reverse=True))
            avg_rca = np.mean(rcas[~np.isinf(rcas)])

            if hparams['voc']:
                print(f"RCA: Avg for {self.work_dir} {gen_dir} vocoder: {avg_rca:.4f}")
            else:
                print(f"RCA: Avg for {self.work_dir} {gen_dir}: {avg_rca:.4f}")

            if hparams['voc']:
                break

    def compute_RRD(self):
        for gen_dir, f_list in sorted(self.f_list.items(), key=lambda x: x[0]):

            results = []
            for i, f_name in tqdm(enumerate(f_list), desc=f'{gen_dir}', ncols=80, leave=False, total=len(f_list)):
                item_name = re.findall(re.compile(r"[\[](.*?)[]]", re.S), f_name)[0]
                spk, song_name, song_idx = item_name.split("#")
                if os.path.exists(f"{hparams['processed_data_dir']}/wav/{spk}#{song_name}#{self.sing_tag}") and \
                        os.path.exists(f"{hparams['processed_data_dir']}/wav/{spk}#{song_name}#{self.sing_tag}/{spk}#{song_name}#{self.sing_tag}#{song_idx}.wav"):
                    g_wav, _ = librosa.core.load(f"{hparams['processed_data_dir']}/wav/{spk}#{song_name}#{self.sing_tag}/{spk}#{song_name}#{self.sing_tag}#{song_idx}.wav",
                                                 sr=hparams.get("audio_sample_rate"))
                else:
                    continue
                p_wav, _ = librosa.core.load(f"{self.work_dir}/{gen_dir}/wavs/{f_name}",
                                             sr=hparams.get("audio_sample_rate"))

                rrd = self.rrd(g_wav, p_wav)
                results.append((f_name[:-4], rrd))

            rrds = np.array([result[1] for result in results])
            rrds = np.array(sorted(rrds, reverse=False))
            avg_rrd = np.mean(rrds[~np.isinf(rrds)])

            if hparams['voc']:
                print(f"RRD: Avg for {self.work_dir} {gen_dir} vocoder: {avg_rrd:.4f}")
            else:
                print(f"RRD: Avg for {self.work_dir} {gen_dir}: {avg_rrd:.4f}")

            if hparams['voc']:
                break

# %%
if __name__ == '__main__':
    hparams = {
        'dataset': 'speech2singing',
        'raw_data_dir': '/data/raw/speech2singing',
        'processed_data_dir': 'data/processed/speech2singing-testdata',
        # 'processed_data_dir': 'data/processed/nhss',
        'binary_data_dir': 'data/binary/speech2singing-testdata',
        'exp_name': 'alignsts_test',
        'lsd': 'lsd2',
        'voc': False,
        'num': 50,

        'fft_size': 512,
        'win_size': 512,
        'hop_size': 128,
        'audio_sample_rate': 24000,
        'fmin': 50,
        'fmax': 11025,
        'audio_num_mel_bins': 80,
        'pitch_extractor': 'parselmouth',
        'spenv_filter': 1.5,
        'sing_spenv_reduced_coef': 400
    }

    evaluation = Evaluation()
    evaluation.compute_LSD()
    evaluation.compute_RCA()
    evaluation.compute_RRD()
