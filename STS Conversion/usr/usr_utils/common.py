import subprocess
import os
import warnings
import re
from collections import OrderedDict
import json

import numpy as np
import parselmouth
import librosa
import pyworld as pw

def exe_cmd(cmd, verbose=True):
    """
    :return: (stdout, stderr=None)
    """
    r = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ret = r.communicate()
    r.stdout.close()
    if verbose:
        res = str(ret[0].decode()).strip()
        if res:
            print(res)
    if ret[1] is not None:
        print(str(ret[0].decode()).strip())
    return ret

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    return path

def note2freq(note, a4=440):
    if type(note) is np.ndarray:
        ret = np.zeros(shape=note.shape)
        tmp = a4 * 2 ** ((note - 69) / 12)
        ret[~np.isclose(note, -1)] = tmp[~np.isclose(note, -1)]
        return ret
    if np.isclose(note, -1):
        return 0.0
    return a4 * 2 ** ((note - 69) / 12)

def freq2note(freq, a4=440):
    if type(freq) is np.ndarray:
        ret = -np.ones(shape=freq.shape)
        tmp = np.log2(freq/a4 + 1e-6) * 12 + 69
        ret[~np.isclose(freq, 0.0)] = tmp[~np.isclose(freq, 0.0)]
        return ret
    if np.isclose(freq, 0.0):
        return -1
    return np.log2(freq / a4) * 12 + 69

def load_audio(f_path, fs=22050, filter_warnings=True):
    if not filter_warnings:
        y, fs = librosa.load(f_path, dtype=float, sr=fs)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y, fs = librosa.load(f_path, dtype=float, sr=fs)
    if len(y.shape) > 1 and y.shape[0] > 1:     # multi-channel
        y = librosa.to_mono(y)
    return y, fs

def get_f0(y, fs, method='parselmouth', f0_min=100, f0_max=800, cutoff=True, hop_size=256):
    """

    Args:
        y:
        fs:
        method:
        f0_min:
        f0_max:
        cutoff:
        hop_size:

    Returns: f0, time sequence, sample rate (for f0 series)

    """
    if len(y) <= 0:
        print('ERROR in get_f0: The size of audio sequence should be positive. Skip error...')
        return None, None, None
    if method == 'world':
        try:
            f0, t = pw.dio(y, fs)
            if cutoff:
                f0[f0 < f0_min] = 0.0
                f0[f0 > f0_max] = 0.0
            return f0, t, 1/(t[1] - t[0])
        except Exception as err:
            print('The following exception occurs:')
            print(err)
            print('continue...')
            return None, None, None
    elif method == 'parselmouth':
        try:
            time_step = hop_size / fs
            snd = parselmouth.Sound(y, sampling_frequency=fs)
            f0 = snd.to_pitch(time_step=time_step, pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
            t = np.arange(0.0, time_step * len(f0), time_step)
            t = t[:len(f0)]  # 可能会有浮点数不精确导致的长度不一
            if cutoff:
                f0[f0 < f0_min] = 0.0
                f0[f0 > f0_max] = 0.0
            return f0, t, fs / hop_size
        except Exception as err:
            print('The following exception occurs:')
            print(err)
            print('continue...')
            return None, None, None
    else:
        print('ERROR in get_f0: please specify the method correctly')
        return None, None, None