# some of the parts are adapted from https://github.com/biggytruck/SpeechSplit2

import os
import pickle
import sys
import numpy as np
import soundfile as sf
from numpy.random import RandomState
import multiprocessing as mp
from tqdm import tqdm
import librosa

from data_gen.singing.speech2singing.data_gen_utils import *
from vocoders.base_vocoder import get_vocoder_cls

from utils.hparams import hparams, set_hparams

DEBUG = False

def NN_interpolation(src, dstH, dstW):
    srcH, srcW = src.shape
    dst = np.zeros((dstH, dstW), dtype=src.dtype)
    for i in range(dstH - 1):
        for j in range(dstW - 1):
            src_x = round(i * (srcH / dstH))
            src_y = round(j * (srcW / dstW))
            dst[i, j] = src[src_x, src_y]
    return dst

def make_spect_f0(target_song_type):
    fs = hparams['audio_sample_rate']
    state_count = 1
    # find spk
    spk_list = list(set([fname.split("#")[0] for fname in os.listdir(f"{hparams['processed_data_dir']}/wav/")]))
    spk_list.sort()
    spenv_dir = f"{hparams['processed_data_dir']}/spenv"
    spmel_mono_dir = f"{hparams['processed_data_dir']}/spmel_mono"
    wav_mono_dir = f"{hparams['processed_data_dir']}/wav_mono"
    wav2vec2_mono_dir = f"{hparams['processed_data_dir']}/wav2vec2_mono"
    mkdir(spenv_dir), mkdir(spmel_mono_dir), mkdir(wav_mono_dir), mkdir(wav2vec2_mono_dir)
    if hparams['binarization_args']['with_wav_mono']:
        mkdir(wav_mono_dir)
    for spk in spk_list:
        if DEBUG:
            pass
        print(f'Generating features for speaker {spk}')
        # find files
        file_list = []
        for dname in os.listdir(f"{hparams['processed_data_dir']}/wav/"):
            spk_, song_name, song_type = dname.split("#")
            if spk_ == spk and song_type == target_song_type:
                for fname in os.listdir(f"{hparams['processed_data_dir']}/wav/{dname}"):
                    file_list.append(f"{hparams['processed_data_dir']}/wav/{dname}/{fname}")
        file_list.sort()
        prng = RandomState(state_count)
        wavs, f0s, sps, aps = [], [], [], []
        for fpath in tqdm(file_list, ncols=80, desc=f"{spk} | computing avg f0"):
            try:
                x, _ = librosa.core.load(fpath, sr=fs)
                if x.shape[0] % 256 == 0:
                    x = np.concatenate((x, np.array([1e-06])), axis=0)
                wav = filter_wav(x, prng, fs)

                # get WORLD analyzer parameters
                f0, sp, ap = get_world_params(wav, fs, hop_size=hparams['hop_size'])
            except:
                wav, f0, sp, ap = None, np.zeros(0), None, None

            wavs.append(wav)
            f0s.append(f0)
            sps.append(sp)
            aps.append(ap)

        # smooth pitch to synthesize monotonic speech
        f0s = average_f0s(f0s, mode='global')
        # prepare for wav2vec
        if hparams['binarization_args'].get('with_speech_wav2vec2_mono', False):
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            model_ctc = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        for idx, (fpath, wav, f0, sp, ap) in tqdm(enumerate(zip(file_list, wavs, f0s, sps, aps)),
                                                  ncols=80, total=len(file_list), desc=f"{spk} | outputting"):
            if wav is None:
                continue
            fname = os.path.basename(fpath)
            spk, song_name, song_type, sen_id = fname[:-4].split("#")
            wav_mono = get_monotonic_wav(wav, f0, sp, ap, fs, hop_size=hparams['hop_size'])
            # spmel = get_spmel(wav)
            # save spenv
            spenv = get_spenv(wav_mono, cutoff=hparams['lifter_cutoff'], fft_length=hparams['fft_size'],
                              hop_length=hparams['hop_size'], audio_num_mel_bins=hparams['audio_num_mel_bins'])
            if np.isnan(spenv).sum() > 0:  # there is nan in spenv
                continue
            mkdir(f"{spenv_dir}/{spk}#{song_name}#{song_type}")
            if not DEBUG:
                np.save(f"{spenv_dir}/{spk}#{song_name}#{song_type}/{spk}#{song_name}#{song_type}#{sen_id}.npy",
                        spenv.astype(np.float32), allow_pickle=False)
            # save spmel_mono
            # spmel_mono = get_spmel(wav_mono)
            _, spmel_mono = get_vocoder_cls(hparams).wav2spec(wav_mono)
            mkdir(f"{spmel_mono_dir}/{spk}#{song_name}#{song_type}")
            if not DEBUG:
                np.save(f"{spmel_mono_dir}/{spk}#{song_name}#{song_type}/{spk}#{song_name}#{song_type}#{sen_id}.npy",
                        spmel_mono.astype(np.float32), allow_pickle=False)
            # save wav_mono
            if hparams['binarization_args']['with_wav_mono'] and not DEBUG:
                mkdir(f"{wav_mono_dir}/{spk}#{song_name}#{song_type}")
                np.save(f"{wav_mono_dir}/{spk}#{song_name}#{song_type}/{spk}#{song_name}#{song_type}#{sen_id}.npy",
                        wav_mono.astype(np.float32), allow_pickle=False)
            # save spmel
            # mkdir(f"{spmel_dir}/{spk}#{song_name}#{song_type}")
            # np.save(f"{spmel_dir}/{spk}#{song_name}#{song_type}/{spk}#{song_name}#{song_type}#{sen_id}.npy",
            #         spmel.astype(np.float32), allow_pickle=False)
            # save wav2vec2 feature
            if hparams['binarization_args'].get('with_speech_wav2vec2_mono', False):
                # the pretrained model used sr=16000, need to resample
                wav_mono = librosa.resample(wav_mono, orig_sr=fs, target_sr=16000)
                input_values = processor(wav_mono, sampling_rate=16000, return_tensors='pt').input_values
                logits_ctc = model_ctc(input_values).logits  # torch.Size([1, 97, 32])
                logits_ctc = logits_ctc[0, :, :].detach().numpy()
                # logits_ctc = NN_interpolation(logits_ctc, spenv.shape[0], logits_ctc.shape[1])  # resample to mel size
                mkdir(f"{wav2vec2_mono_dir}/{spk}#{song_name}#{song_type}")
                if not DEBUG:
                    np.save(
                        f"{wav2vec2_mono_dir}/{spk}#{song_name}#{song_type}/{spk}#{song_name}#{song_type}#{sen_id}.npy",
                        logits_ctc.astype(np.float32), allow_pickle=False)


def make_spect_f0_mp(target_song_type):
    fs = hparams['audio_sample_rate']
    state_count = 1
    # find spk
    spk_list = list(set([fname.split("#")[0] for fname in os.listdir(f"{hparams['processed_data_dir']}/wav/")]))
    spmel_dir = f"{hparams['processed_data_dir']}/spmel"
    spenv_dir = f"{hparams['processed_data_dir']}/spenv"
    spmel_mono_dir = f"{hparams['processed_data_dir']}/spmel_mono"
    wav_mono_dir = f"{hparams['processed_data_dir']}/wav_mono"
    wav2vec2_mono_dir = f"{hparams['processed_data_dir']}/wav2vec2_mono"
    mkdir(spmel_dir), mkdir(spenv_dir), mkdir(spmel_mono_dir), mkdir(wav2vec2_mono_dir)
    if hparams['binarization_args']['with_wav_mono']:
        mkdir(wav_mono_dir)

    num_task = len(spk_list)
    pool = mp.Pool(min(num_task, 40))
    for task_idx, spk in enumerate(spk_list):
        pool.apply_async(make_spect_f0_slave,
                         args=(task_idx, spk, wav_mono_dir,
                               spmel_dir, spenv_dir, spmel_mono_dir, wav2vec2_mono_dir, target_song_type,
                               state_count, fs, num_task))
    pool.close()
    pool.join()

def make_spect_f0_slave(task_idx, spk, wav_mono_dir,
                               spmel_dir, spenv_dir, spmel_mono_dir, wav2vec2_mono_dir, target_song_type, state_count,
                        fs, num_task):
    print(f'Generating features for speaker {spk}')
    # find files
    file_list = []
    for dname in os.listdir(f"{hparams['processed_data_dir']}/wav/"):
        spk_, song_name, song_type = dname.split("#")
        if spk_ == spk and song_type == target_song_type:
            for fname in os.listdir(f"{hparams['processed_data_dir']}/wav/{dname}"):
                file_list.append(f"{hparams['processed_data_dir']}/wav/{dname}/{fname}")
    file_list.sort()
    prng = RandomState(state_count)
    wavs, f0s, sps, aps = [], [], [], []
    for fpath in tqdm(file_list, ncols=80, desc=f"{spk} | computing avg f0", position=task_idx):
        try:
            x, _ = librosa.core.load(fpath, sr=fs)
            if x.shape[0] % 256 == 0:
                x = np.concatenate((x, np.array([1e-06])), axis=0)
            wav = filter_wav(x, prng, fs)

            # get WORLD analyzer parameters
            f0, sp, ap = get_world_params(wav, fs, hop_size=hparams['hop_size'])
        except:
            wav, f0, sp, ap = None, np.zeros(0), None, None

        wavs.append(wav)
        f0s.append(f0)
        sps.append(sp)
        aps.append(ap)

    # smooth pitch to synthesize monotonic speech
    f0s = average_f0s(f0s, mode='global')
    # prepare for wav2vec
    if hparams['binarization_args'].get('with_speech_wav2vec2_mono', False):
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model_ctc = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    for idx, (fpath, wav, f0, sp, ap) in tqdm(enumerate(zip(file_list, wavs, f0s, sps, aps)), ncols=80,
                                              total=len(file_list), desc=f"{spk} | outputting",
                                              position=num_task + task_idx):
        if wav is None:
            continue
        fname = os.path.basename(fpath)
        spk, song_name, song_type, sen_id = fname[:-4].split("#")
        wav_mono = get_monotonic_wav(wav, f0, sp, ap, fs, hop_size=hparams['hop_size'])
        # spmel = get_spmel(wav)    # no need for mel
        # save spenv
        spenv = get_spenv(wav_mono, cutoff=hparams['lifter_cutoff'], fft_length=hparams['fft_size'],
                              hop_length=hparams['hop_size'], audio_num_mel_bins=hparams['audio_num_mel_bins'])
        if np.isnan(spenv).sum() > 0:  # there is nan in spenv
            continue
        mkdir(f"{spenv_dir}/{spk}#{song_name}#{song_type}")
        np.save(f"{spenv_dir}/{spk}#{song_name}#{song_type}/{spk}#{song_name}#{song_type}#{sen_id}.npy",
                spenv.astype(np.float32), allow_pickle=False)
        # save spmel_mono
        # spmel_mono = get_spmel(wav_mono)
        _, spmel_mono = get_vocoder_cls(hparams).wav2spec(wav_mono)
        mkdir(f"{spmel_mono_dir}/{spk}#{song_name}#{song_type}")
        np.save(f"{spmel_mono_dir}/{spk}#{song_name}#{song_type}/{spk}#{song_name}#{song_type}#{sen_id}.npy",
                spmel_mono.astype(np.float32), allow_pickle=False)
        # save wav_mono
        if hparams['binarization_args']['with_wav_mono']:
            mkdir(f"{wav_mono_dir}/{spk}#{song_name}#{song_type}")
            np.save(f"{wav_mono_dir}/{spk}#{song_name}#{song_type}/{spk}#{song_name}#{song_type}#{sen_id}.npy",
                    wav_mono.astype(np.float32), allow_pickle=False)
        # save spmel
        # mkdir(f"{spmel_dir}/{spk}#{song_name}#{song_type}")
        # np.save(f"{spmel_dir}/{spk}#{song_name}#{song_type}/{spk}#{song_name}#{song_type}#{sen_id}.npy",
        #         spmel.astype(np.float32), allow_pickle=False)
        # save wav2vec2 feature
        if hparams['binarization_args'].get('with_speech_wav2vec2_mono', False):
            # the pretrained model used sr=16000, may need to resample
            wav_mono = librosa.resample(wav_mono, orig_sr=fs, target_sr=16000)
            input_values = processor(wav_mono, sampling_rate=16000, return_tensors='pt').input_values
            logits_ctc = model_ctc(input_values).logits  # torch.Size([1, 97, 32])
            logits_ctc = logits_ctc[0, :, :].detach().numpy()
            # logits_ctc = NN_interpolation(logits_ctc, spenv.shape[0], logits_ctc.shape[1])  # resample to mel size
            mkdir(f"{wav2vec2_mono_dir}/{spk}#{song_name}#{song_type}")
            np.save(f"{wav2vec2_mono_dir}/{spk}#{song_name}#{song_type}/{spk}#{song_name}#{song_type}#{sen_id}.npy",
                    logits_ctc.astype(np.float32), allow_pickle=False)

def preprocess_data():
    print('Start preprocessing...')
    # make_spect_f0()
    make_spect_f0_mp()
    # make_metadata()
    print('Done')


if __name__ == '__main__':
    set_hparams()
    # set_hparams('configs/singing/speech2singing/alignsts.yaml')
    if DEBUG:
        make_spect_f0(target_song_type='歌词')
        if hparams['binarization_args'].get('with_sing_spenv', False):
            make_spect_f0(target_song_type='专业')
        # make_spect_f0(target_song_type='speech')
        # if hparams['binarization_args'].get('with_sing_spenv', False):
        #     make_spect_f0(target_song_type='sing')
    else:
        make_spect_f0_mp(target_song_type='歌词')
        if hparams['binarization_args'].get('with_sing_spenv', False):
            make_spect_f0_mp(target_song_type='专业')
    # make_spect_f0(target_song_type='歌词')
    # if hparams['binarization_args'].get('with_sing_spenv', False):
    #     make_spect_f0(target_song_type='专业')
