import librosa
import numpy as np
import wave
import soundfile as sf

from utils.audio.vad import trim_long_silences


def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return 10.0 ** (x * 0.05)


def normalize(S, min_level_db):
    return (S - min_level_db) / -min_level_db


def denormalize(D, min_level_db):
    return (D * -min_level_db) + min_level_db


def librosa_wav2spec(wav_path,
                     fft_size=1024,
                     hop_size=256,
                     win_length=1024,
                     window="hann",
                     num_mels=80,
                     fmin=80,
                     fmax=-1,
                     eps=1e-6,
                     sample_rate=22050,
                     loud_norm=False,
                     trim_long_sil=False):
    import pyloudnorm as pyln
    if isinstance(wav_path, str):
        if trim_long_sil:
            wav, _, _ = trim_long_silences(wav_path, sample_rate)
        else:
            wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path
    wav_orig = np.copy(wav)

    if loud_norm:
        meter = pyln.Meter(sample_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -22.0)
        if np.abs(wav).max() > 1:
            wav = wav / np.abs(wav).max()

    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="constant")
    linear_spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)

    # calculate mel spec
    mel = mel_basis @ linear_spc
    mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    l_pad, r_pad = librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
    wav = wav[:mel.shape[1] * hop_size]

    # log linear spec
    linear_spc = np.log10(np.maximum(eps, linear_spc))
    return {'wav': wav, 'mel': mel.T, 'linear': linear_spc.T, 'mel_basis': mel_basis, 'wav_orig': wav_orig}

def get_wav_num_frames(path, sr=None):
    try:
        with wave.open(path, 'rb') as f:
            sr_ = f.getframerate()
            if sr is None:
                sr = sr_
            return int(f.getnframes() / (sr_ / sr))
    except wave.Error:
        wav_file, sr_ = sf.read(path, dtype='float32')
        if sr is None:
                sr = sr_
        return int(len(wav_file) / (sr_ / sr))
    except:
        wav_file, sr_ = librosa.core.load(path, sr=sr)
        return len(wav_file)


def pad_frames(frames, hop_size, n_samples, n_expect):
    n_frames = frames.shape[0]
    lpad = (int(n_samples // hop_size) - n_frames + 1) // 2
    rpad = n_expect - n_frames - lpad
    if rpad < 0:
        frames = frames[:rpad]
        rpad = 0
    if lpad > 0 or rpad > 0:
        frames = np.pad(frames, [[lpad, rpad]], mode='constant')
    return frames


def get_zcr_librosa(wav_data, length, hparams):
    import librosa.feature
    hop_size = hparams['hop_size']
    win_size = hparams['win_size']

    zcr = librosa.feature.zero_crossing_rate(wav_data, frame_length=win_size, hop_length=hop_size)[0]
    zcr = pad_frames(zcr, hop_size, wav_data.shape[0], length)
    return zcr


def get_energy_librosa(wav_data, length, hparams):
    """

    :param wav_data: [T]
    :param length: Expected number of frames
    :param hparams:
    :return: energy
    """
    import librosa.feature
    hop_size = hparams['hop_size']
    win_size = hparams['win_size']

    energy = librosa.feature.rms(y=wav_data, frame_length=win_size, hop_length=hop_size)[0]
    energy = pad_frames(energy, hop_size, wav_data.shape[0], length)
    return energy


def get_breathiness_pyworld(wav_data, f0, length, hparams):
    """

    :param wav_data: [T]
    :param f0: reference f0
    :param length: Expected number of frames
    :param hparams:
    :return: breathiness
    """
    import pyworld as pw
    sample_rate = hparams['audio_sample_rate']
    hop_size = hparams['hop_size']
    fft_size = hparams['fft_size']

    x = wav_data.astype(np.double)
    f0 = f0.astype(np.double)
    wav_frames = (x.shape[0] + hop_size - 1) // hop_size
    f0_frames = f0.shape[0]
    if f0_frames < wav_frames:
        f0 = np.pad(f0, [[0, wav_frames - f0_frames]], mode='constant')
    elif f0_frames > wav_frames:
        f0 = f0[:wav_frames]

    time_step = hop_size / sample_rate
    t = np.arange(0, wav_frames) * time_step
    sp = pw.cheaptrick(x, f0, t, sample_rate, fft_size=fft_size)  # extract smoothed spectrogram
    ap = pw.d4c(x, f0, t, sample_rate, fft_size=fft_size)  # extract aperiodicity
    y = pw.synthesize(
        f0, sp * ap * ap, np.ones_like(ap), sample_rate,
        frame_period=time_step * 1000
    )  # synthesize the aperiodic part using the parameters
    breathiness = get_energy_librosa(y, length, hparams)
    breathiness = breathiness.astype(float)
    return breathiness

def norm_energy(energy, norm='log'):
    import torch
    is_torch = isinstance(energy, torch.Tensor)
    if norm == 'log':
        energy = torch.log10(energy + 1e-6) if is_torch else np.log10(energy + 1e-6)
    return energy

def denorm_energy(energy, norm='log', energy_padding=None, min=0, max=0.8):
    import torch
    is_torch = isinstance(energy, torch.Tensor)
    if norm == 'log':
        energy = 10 ** energy - 1e-6
    energy = energy.clamp(min=min, max=max) if is_torch else np.clip(energy, a_min=min, a_max=max)
    if energy_padding is not None:
        energy[energy_padding] = 0
    return energy

def energy_to_coarse(energy, energy_bin=256, energy_max=0.8, energy_min=1e-6):
    import torch
    is_torch = isinstance(energy, torch.Tensor)
    energy_max = np.log10(energy_max)
    energy_min = np.log10(energy_min)
    energy = norm_energy(energy)
    energy = energy.clamp(min=energy_min, max=energy_max) \
        if is_torch else np.clip(energy, a_min=energy_min, a_max=energy_max)

    energy[energy != 0] = (energy[energy != 0] - energy_min) * (energy_bin - 2) / (energy_max - energy_min) + 1
    energy[energy <= 1] = 1
    energy[energy > energy_bin - 1] = energy_bin - 1
    energy_coarse = (energy + 0.5).long() if is_torch else np.rint(energy).astype(int)
    assert energy_coarse.max() <= energy_bin-1 and energy_coarse.min() >= 1, \
        (energy_coarse.max(), energy_coarse.min(), energy.max(), energy.min())

    return energy_coarse

def coarse_to_energy(energy_coarse, energy_bin=256, energy_max=0.8, energy_min=1e-6):
    energy_max = np.log10(energy_max)
    energy_min = np.log10(energy_min)
    zeros = energy_coarse == 1
    energy = energy_min + (energy_coarse - 1) * (energy_max - energy_min) / (energy_bin - 2)
    energy = 10 ** energy - 1e-6
    energy[zeros] = 0
    return energy

def anything_to_coarse(x, bins=256, x_max=1., x_min=0., pad=0):
    import torch
    is_torch = isinstance(x, torch.Tensor)
    x = x.clamp(min=x_min, max=x_max) if is_torch else np.clip(x, a_min=x_min, a_max=x_max)

    x[x != pad] = (x[x != pad] - x_min) * (bins - 2) / (x_max - x_min) + 1
    x[x <= 1] = 1
    x[x > bins - 1] = bins - 1
    x_coarse = (x + 0.5).long() if is_torch else np.rint(x).astype(int)
    assert x_coarse.max() <= bins - 1 and x_coarse.min() >= 1, (x_coarse.max(), x_coarse.min(), x.max(), x.min())

    return x_coarse

def coarse_to_anything(x_coarse, bins=256, x_max=1., x_min=0.):
    zeros = x_coarse == 1
    x = x_min + (x_coarse - 1) * (x_max - x_min) / (bins - 2)
    x[zeros] = 0
    return x
