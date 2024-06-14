import math
import numpy as np

PITCH_EXTRACTOR = {}


def register_pitch_extractor(name):
    def register_pitch_extractor_(cls):
        PITCH_EXTRACTOR[name] = cls
        return cls

    return register_pitch_extractor_


def get_pitch_extractor(name):
    return PITCH_EXTRACTOR[name]


def extract_pitch_simple(wav):
    from utils.commons.hparams import hparams
    return extract_pitch(hparams['pitch_extractor'], wav,
                         hparams['hop_size'], hparams['audio_sample_rate'],
                         f0_min=hparams['f0_min'], f0_max=hparams['f0_max'])


def extract_pitch(extractor_name, wav_data, hop_size, audio_sample_rate, f0_min=75, f0_max=800, **kwargs):
    return get_pitch_extractor(extractor_name)(wav_data, hop_size, audio_sample_rate, f0_min, f0_max, **kwargs)


@register_pitch_extractor('parselmouth')
def parselmouth_pitch(wav_data, hop_size, audio_sample_rate, f0_min, f0_max,
                      voicing_threshold=0.6, *args, **kwargs):
    import parselmouth
    time_step = hop_size / audio_sample_rate * 1000
    n_mel_frames = int(len(wav_data) // hop_size)
    f0_pm = parselmouth.Sound(wav_data, audio_sample_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=voicing_threshold,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    pad_size = (n_mel_frames - len(f0_pm) + 1) // 2
    f0 = np.pad(f0_pm, [[pad_size, n_mel_frames - len(f0_pm) - pad_size]], mode='constant')
    return f0

@register_pitch_extractor('pyworld')
def pyworld_pitch(wav_data, hop_size, audio_sample_rate, f0_min, f0_max,
                      voicing_threshold=0.6, *args, **kwargs):
    import pyworld as pw
    # f0, _ = pw.harvest(wav_data.astype(np.double), audio_sample_rate, f0_floor=f0_min, f0_ceil=f0_max,
    #                    frame_period=hop_size * 1000 / audio_sample_rate)
    f0, _ = pw.dio(wav_data.astype(np.double), audio_sample_rate, f0_floor=f0_min, f0_ceil=f0_max, frame_period=hop_size * 1000 / audio_sample_rate)
    f0[f0 < f0_min] = 0.0
    f0[f0 > f0_max] = 0.0
    n_mel_frames = math.ceil(len(wav_data) / hop_size)
    if n_mel_frames > len(f0):
        pad_size = (n_mel_frames - len(f0) + 1) // 2
        f0 = np.pad(f0, [[pad_size, n_mel_frames - len(f0) - pad_size]], mode='constant')
    elif n_mel_frames < len(f0):
        left_del = (len(f0) - n_mel_frames + 1) // 2
        right_del = len(f0) - n_mel_frames - left_del
        f0 = f0[left_del: (-right_del if right_del > 0 else len(f0))]
    return f0
