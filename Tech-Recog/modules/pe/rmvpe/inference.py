import math

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
import pyworld as pw

from utils.audio.pitch_utils import interp_f0, resample_align_curve
from .constants import *
from .model import E2E0
from .spec import MelSpectrogram
from .utils import to_local_average_f0, to_viterbi_f0


class RMVPE:
    def __init__(self, model_path, hop_length=160, device=None):
        self.resample_kernel = {}
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model = E2E0(4, 1, (2, 2)).eval().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX
        ).to(self.device)
        self.hop_length = hop_length

    @torch.no_grad()
    def mel2hidden(self, mel):
        n_frames = mel.shape[-1]
        mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode='constant')
        hidden = self.model(mel)
        return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        if use_viterbi:
            f0 = to_viterbi_f0(hidden, thred=thred)
        else:
            f0 = to_local_average_f0(hidden, thred=thred)
        return f0

    def postprocess(self, f0, fmin=50, fmax=1000, audio=None, min_gap=2):
        if audio is not None:
            # this doesn't work. deprecated
            t = np.arange(0, f0.shape[0] * self.hop_length / 16000, self.hop_length / 16000)
            f0 = pw.stonemask(audio.astype(np.float64), f0.astype(np.float64), t, 16000).astype(float)
        f0[f0 < fmin] = 0
        f0[f0 > fmax] = 0
        # eliminate glitch
        # min_gap: if successive positive f0 positions < min_gap, zero these positions
        # eg: if min_gap=2, [0, 500, 500, 0] => [0, 0, 0, 0]
        for idx in range(f0.shape[0] - min_gap - 1):
            if f0[idx] == 0 and f0[idx + min_gap + 1] == 0 and np.sum(f0[idx: idx + min_gap + 2]) > 0:
                f0[idx: idx + min_gap + 2] = 0
        return f0

    def infer_from_audio(self, audio, sample_rate=16000, thred=0.03, use_viterbi=False):
        audio = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        if sample_rate == 16000:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, 16000, lowpass_filter_width=128)
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
        mel = self.mel_extractor(audio_res, center=True)
        hidden = self.mel2hidden(mel)
        f0 = self.decode(hidden, thred=thred, use_viterbi=use_viterbi).squeeze(0)
        return f0

    def get_pitch(self, waveform, sample_rate, hop_size, length, interp_uv=False, fmin=50, fmax=1000):
        f0 = self.infer_from_audio(waveform, sample_rate=sample_rate)
        f0 = self.postprocess(f0, fmin, fmax)
        uv = f0 == 0
        time_step = hop_size / sample_rate
        f0_res = resample_align_curve(f0, 0.01, time_step, length)
        uv_res = resample_align_curve(uv.astype(np.float32), 0.01, time_step, length) > 0.5
        if not interp_uv:
            f0_res[uv_res] = 0
        return f0_res, uv_res

    def infer_from_audio_batch(self, audios, sample_rate=16000, thred=0.03, use_viterbi=False):
        from utils.commons.dataset_utils import collate_1d_or_2d
        audios = [torch.from_numpy(audio).float() for audio in audios]
        sizes = [math.ceil((audio.shape[0] + 1) / self.hop_length) for audio in audios]
        audios = collate_1d_or_2d(audios, 0.0).to(self.device)
        if sample_rate == 16000:
            audios_res = audios
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, 16000, lowpass_filter_width=128)
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(self.device)
            audios_res = self.resample_kernel[key_str](audios)
        mels = self.mel_extractor(audios_res, center=True)
        hiddens = self.mel2hidden(mels)
        f0 = self.decode(hiddens, thred=thred, use_viterbi=use_viterbi)
        f0s = []
        for i in range(f0.shape[0]):
            f = f0[i, :sizes[i]]
            f0s.append(f)
        return f0s

    def get_pitch_batch(self, waveforms, sample_rate, hop_size, lengths, interp_uv=False, fmin=50, fmax=1000):
        # hop_size, sample_rate: tgt params
        f0s = self.infer_from_audio_batch(waveforms, sample_rate=sample_rate)
        f0s_res, uvs_res = [], []
        for idx, f0 in enumerate(f0s):
            f0 = self.postprocess(f0, fmin, fmax, min_gap=6)
            uv = f0 == 0
            length = lengths[idx]
            time_step = hop_size / sample_rate
            f0_res = resample_align_curve(f0, 0.01, time_step, length)
            uv_res = resample_align_curve(uv.astype(np.float32), 0.01, time_step, length) > 0.5
            if not interp_uv:
                f0_res[uv_res] = 0
            f0s_res.append(f0_res)
            uvs_res.append(uv_res)
        return f0s_res, uvs_res

    def release_cuda(self):
        self.model = self.model.cpu()
        self.mel_extractor = self.mel_extractor.cpu()
