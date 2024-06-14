import numpy as np
import torch
import pretty_midi

def to_lf0(f0):
    f0[f0 < 1.0e-5] = 1.0e-6
    lf0 = f0.log() if isinstance(f0, torch.Tensor) else np.log(f0)
    lf0[f0 < 1.0e-5] = - 1.0E+10
    return lf0


def to_f0(lf0):
    f0 = np.where(lf0 <= 0, 0.0, np.exp(lf0))
    return f0.flatten()


def f0_to_coarse(f0, f0_bin=256, f0_max=900.0, f0_min=50.0):
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(int)
    assert f0_coarse.max() <= f0_bin-1 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min(), f0.min(), f0.max())
    return f0_coarse


def coarse_to_f0(f0_coarse, f0_bin=256, f0_max=900.0, f0_min=50.0):
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    uv = f0_coarse == 1
    f0 = f0_mel_min + (f0_coarse - 1) * (f0_mel_max - f0_mel_min) / (f0_bin - 2)
    f0 = ((f0 / 1127).exp() - 1) * 700
    f0[uv] = 0
    return f0


def norm_f0(f0, uv, pitch_norm='log', f0_mean=400, f0_std=100):
    is_torch = isinstance(f0, torch.Tensor)
    if pitch_norm == 'standard':
        f0 = (f0 - f0_mean) / f0_std
    if pitch_norm == 'log':
        f0 = torch.log2(f0 + 1e-8) if is_torch else np.log2(f0 + 1e-8)
    if uv is not None:
        f0[uv > 0] = 0
    return f0


def norm_interp_f0(f0, pitch_norm='log', f0_mean=None, f0_std=None):
    is_torch = isinstance(f0, torch.Tensor)
    if is_torch:
        device = f0.device
        f0 = f0.data.cpu().numpy()
    uv = f0 == 0
    f0 = norm_f0(f0, uv, pitch_norm, f0_mean, f0_std)
    if sum(uv) == len(f0):
        f0[uv] = 0
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    if is_torch:
        uv = torch.FloatTensor(uv)
        f0 = torch.FloatTensor(f0)
        f0 = f0.to(device)
        uv = uv.to(device)
    return f0, uv


def denorm_f0(f0, uv, pitch_norm='log', f0_mean=400, f0_std=100, pitch_padding=None, min=50, max=900):
    is_torch = isinstance(f0, torch.Tensor)
    if pitch_norm == 'standard':
        f0 = f0 * f0_std + f0_mean
    if pitch_norm == 'log':
        f0 = 2 ** f0
    f0 = f0.clamp(min=min, max=max) if is_torch else np.clip(f0, a_min=min, a_max=max)
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0

def interp_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = norm_f0(f0, uv)
    if uv.any() and not uv.all():
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return denorm_f0(f0, uv=None), uv

def resample_align_curve(points: np.ndarray, original_timestep: float, target_timestep: float, align_length: int):
    t_max = (len(points) - 1) * original_timestep
    curve_interp = np.interp(
        np.arange(0, t_max, target_timestep),
        original_timestep * np.arange(len(points)),
        points
    ).astype(points.dtype)
    delta_l = align_length - len(curve_interp)
    if delta_l < 0:
        curve_interp = curve_interp[:align_length]
    elif delta_l > 0:
        curve_interp = np.concatenate((curve_interp, np.full(delta_l, fill_value=curve_interp[-1])), axis=0)
    return curve_interp

def midi_to_hz(midi):
    if type(midi) == np.ndarray:
        non_mask = midi == 0
        freq_hz = 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
        freq_hz[non_mask] = 0
    else:
        freq_hz = 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
    return freq_hz

def hz_to_midi(hz):
    if type(hz) == torch.Tensor:
        non_mask = hz == 0
        midi = 69.0 + 12.0 * (torch.log2(hz) - torch.log2(torch.Tensor(440.0)))
        midi[non_mask] = 0
    elif type(hz) == np.ndarray:
        non_mask = hz == 0
        midi = 69.0 + 12.0 * (np.log2(hz) - np.log2(440.0))
        midi[non_mask] = 0
    else:
        midi = 69.0 + 12.0 * (np.log2(hz) - np.log2(440.0))
        if hz == 0:
            midi = 0
    return midi

def boundary2Interval(bd):
    # bd has a shape of [T] with T frames
    is_torch = isinstance(bd, torch.Tensor)
    if is_torch:
        device = bd.device
        bd = bd.data.cpu().numpy()
    assert len(bd.shape) == 1
    # force valid begin and end
    # bd[0] = 0     # took care of in regulate_boundary()
    # bd[-1] = 0
    ret = np.zeros(shape=(bd.sum() + 1, 2), dtype=int)
    ret_idx = 0
    ret[0, 0] = 0
    for i, u in enumerate(bd):
        if i == 0:
            continue
        if u == 1:
            ret[ret_idx, 1] = i
            ret[ret_idx+1, 0] = i
            ret_idx += 1
    ret[-1, 1] = bd.shape[0] - 1
    if is_torch:
        ret = torch.LongTensor(ret).to(device)
    return ret

def validate_pitch_and_itv(notes, note_itv):
    # notes [T]
    # note_itv [T, 2]
    assert notes.shape[0] == note_itv.shape[0]
    res_notes = []
    res_note_itv = []
    for idx in range(notes.shape[0]):
        pitch, itv = notes[idx], note_itv[idx]
        if itv[0] >= itv[1]:
            # for i in range(10):
            #     print()
            #     print('-'*30)
            #     print(note_itv)
            raise RuntimeError("The note duration should be positive")
        if pitch == 0:
            continue
        res_notes.append(pitch)
        res_note_itv.append([itv[0], itv[1]])
    res_notes = np.array(res_notes)
    res_note_itv = np.array(res_note_itv)
    return res_notes, res_note_itv

def save_midi(notes, note_itv, midi_path):
    # notes [T]
    # note_itv [T, 2]
    notes, note_itv = validate_pitch_and_itv(notes, note_itv)
    if notes.shape == (0,):
        return None
    assert notes.shape[0] == note_itv.shape[0]
    piano_chord = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    for idx in range(notes.shape[0]):
        pitch, itv = notes[idx], note_itv[idx]
        note = pretty_midi.Note(velocity=120, pitch=pitch, start=itv[0], end=itv[1])
        piano.notes.append(note)
    piano_chord.remove_invalid_notes()
    piano_chord.instruments.append(piano)
    piano_chord.write(midi_path)
    return piano_chord

def midi2NoteInterval(mid):
    assert type(mid) == pretty_midi.PrettyMIDI
    if len(mid.instruments) == 0 or len(mid.instruments[0].notes) == 0:
        return None
    ret = np.zeros(shape=(len(mid.instruments[0].notes), 2))
    for i, note in enumerate(mid.instruments[0].notes):
        ret[i, 0] = note.start
        ret[i, 1] = note.end
    return ret

def midi2NotePitch(mid):
    assert type(mid) == pretty_midi.PrettyMIDI
    if len(mid.instruments) == 0 or len(mid.instruments[0].notes) == 0:
        return None
    ret = np.zeros(shape=len(mid.instruments[0].notes))
    for i, note in enumerate(mid.instruments[0].notes):
        ret[i] = note.pitch
    return ret

def midi_onset_eval(mid_gt, mid_pred):
    import mir_eval
    interval_true = midi2NoteInterval(mid_gt)
    if interval_true is None:
        raise RuntimeError('Midi ground truth is None')
    interval_pred = midi2NoteInterval(mid_pred)
    if interval_pred is None:
        return 0, 0, 0
    onset_p, onset_r, onset_f = mir_eval.transcription.onset_precision_recall_f1(
        interval_true, interval_pred, onset_tolerance=0.05, strict=False, beta=1.0)
    return onset_p, onset_r, onset_f

def midi_offset_eval(mid_gt, mid_pred):
    import mir_eval
    interval_true = midi2NoteInterval(mid_gt)
    if interval_true is None:
        raise RuntimeError('Midi ground truth is None')
    interval_pred = midi2NoteInterval(mid_pred)
    if interval_pred is None:
        return 0, 0, 0
    offset_p, offset_r, offset_f = mir_eval.transcription.offset_precision_recall_f1(
        interval_true, interval_pred, offset_ratio=0.2, offset_min_tolerance=0.05, strict=False, beta=1.0)
    return offset_p, offset_r, offset_f

def midi_pitch_eval(mid_gt, mid_pred):
    import mir_eval
    interval_true = midi2NoteInterval(mid_gt)
    pitch_true = midi_to_hz(midi2NotePitch(mid_gt))
    if interval_true is None or pitch_true is None:
        raise RuntimeError('Midi ground truth is None')
    interval_pred = midi2NoteInterval(mid_pred)
    pitch_pred = midi2NotePitch(mid_pred)
    if interval_pred is None:
        return 0, 0, 0, 0
    if pitch_pred is None:
        pitch_pred = np.zeros(interval_pred.shape[0])
    pitch_pred = midi_to_hz(pitch_pred)
    overlap_p, overlap_r, overlap_f, avg_overlap_ratio = mir_eval.transcription.precision_recall_f1_overlap(
        interval_true, pitch_true, interval_pred, pitch_pred, onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=0.2, offset_min_tolerance=0.05, strict=False, beta=1.0)
    return overlap_p, overlap_r, overlap_f, avg_overlap_ratio

def midi_melody_eval(mid_gt, mid_pred, hop_size=256, sample_rate=48000):
    interval_true = midi2NoteInterval(mid_gt)
    pitch_true = midi_to_hz(midi2NotePitch(mid_gt))
    if interval_true is None or pitch_true is None:
        raise RuntimeError('Midi ground truth is None')
    interval_pred = midi2NoteInterval(mid_pred)
    pitch_pred = midi2NotePitch(mid_pred)
    if interval_pred is None:
        return 0, 0, 0, 0
    if pitch_pred is None:
        pitch_pred = np.zeros(interval_pred.shape[0])
    pitch_pred = midi_to_hz(pitch_pred)

    vr, vfa, rpa, rca, oa = melody_eval_pitch_and_itv(
        pitch_true, interval_true, pitch_pred, interval_pred, hop_size, sample_rate)

    return vr, vfa, rpa, rca, oa

def melody_eval_pitch_and_itv(pitch_true, interval_true, pitch_pred, interval_pred, hop_size=256, sample_rate=48000):
    import mir_eval
    t_gt = np.arange(0, interval_true[-1][1], hop_size / sample_rate)
    freq_gt = np.zeros_like(t_gt)
    for idx in range(len(pitch_true)):
        freq_gt[min(len(freq_gt) - 1, round(interval_true[idx][0] * sample_rate / hop_size)): round(
            interval_true[idx][1] * sample_rate / hop_size)] = pitch_true[idx]

    t_pred = np.arange(0, interval_pred[-1][1], hop_size / sample_rate)
    freq_pred = np.zeros_like(t_pred)
    for idx in range(len(pitch_pred)):
        freq_pred[min(len(freq_pred) - 1, round(interval_pred[idx][0] * sample_rate / hop_size)): round(
            interval_pred[idx][1] * sample_rate / hop_size)] = pitch_pred[idx]

    ref_voicing, ref_cent, est_voicing, est_cent = mir_eval.melody.to_cent_voicing(t_gt, freq_gt,
                                                                                   t_pred, freq_pred)
    vr, vfa = mir_eval.melody.voicing_measures(ref_voicing,
                                               est_voicing)  # voicing recall, voicing false alarm
    rpa = mir_eval.melody.raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
    rca = mir_eval.melody.raw_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
    oa = mir_eval.melody.overall_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)

    return vr, vfa, rpa, rca, oa

