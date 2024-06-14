import os
import math
import gc

import librosa.feature
import numpy as np
import torch
from tqdm import tqdm

from tasks.tts.dataset_utils import BaseSpeechDataset
from utils.commons.dataset_utils import collate_1d_or_2d, pad_or_cut_xd
from utils.commons.indexed_datasets import IndexedDataset
from utils.audio import librosa_wav2spec, energy_to_coarse, coarse_to_energy, anything_to_coarse, coarse_to_anything
from utils.audio.pitch_utils import norm_interp_f0, denorm_f0, f0_to_coarse
from utils.commons.signal import get_filter_1d, get_gaussian_kernel_1d, get_hann_kernel_1d, \
    get_triangle_kernel_1d, add_gaussian_noise

def get_soft_label_filter(soft_label_func, win_size, hparams):
    # win_size: ms
    win_size = round(int(win_size) * hparams['audio_sample_rate'] / 1000 / hparams['hop_size'])
    win_size = win_size if win_size % 2 == 1 else win_size + 1  # ensure odd number
    if soft_label_func == 'gaussian':
        sigma = win_size / 3 / 2  # 3sigma range
        kernel = get_gaussian_kernel_1d(win_size, sigma)
        kernel = kernel / kernel.max()  # make sure the middle is 1
    elif soft_label_func == 'hann':
        kernel = get_hann_kernel_1d(win_size, periodic=False)
    elif soft_label_func == 'triangle':
        kernel = get_triangle_kernel_1d(win_size)
    soft_filter = get_filter_1d(kernel, win_size, channels=1)
    return soft_filter

def add_plain_noise(x, type_and_std=None, random_std=True, x_min=None, x_max=None):
    if type_and_std in ['none', None]:
        return x
    noise_type, std = type_and_std.split(':')
    if noise_type == 'gaussian':
        std = float(std) * np.random.rand() if random_std else float(std)
        x = add_gaussian_noise(x, mean=0.0, std=std)
    if x_min is not None and x_max is not None:
        x = x.clamp(min=x_min, max=x_max)
    return x

class TEDataset(BaseSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super(TEDataset, self).__init__(prefix, shuffle, items, data_dir)
        self.soft_filter = {}
        # self.noise_ds = IndexedDataset(f"{self.hparams['noise_data_dir']}/{prefix}")  # error in multiprocessing
        self.noise_ds = None
        noise_snr = self.hparams.get('noise_snr', '6-20')
        if '-' in noise_snr:
            l, r = noise_snr.split('-')
            self.noise_snr = (float(l), float(r))
        else:
            self.noise_snr = float(noise_snr)
        if items is None and self.avail_idxs is not None:
            ds_names_selected = None
            if prefix in ['train', 'valid'] and self.hparams.get('ds_names_in_training', '') != '':
                ds_names_selected = self.hparams.get('ds_names_in_training', '').split(';') + ['']
                print(f'| Iterating training sets to find samples belong to datasets {ds_names_selected[:-1]}')
            elif prefix == 'test' and self.hparams.get('ds_names_in_testing', '') != '':
                ds_names_selected = self.hparams.get('ds_names_in_testing', '').split(';') + ['']
                print(f'| Iterating testing sets to find samples belong to datasets {ds_names_selected[:-1]}')
            if ds_names_selected is not None:
                avail_idxs = []
                # somehow, can't use self.indexed_ds beforehand (need to create a temp), otherwise '_pickle.UnpicklingError'
                temp_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
                for idx in tqdm(range(len(self)), total=len(self)):
                    item = temp_ds[self.avail_idxs[idx]]
                    if item.get('ds_name', '') in ds_names_selected:
                        avail_idxs.append(self.avail_idxs[idx])
                print(f'| Chose [{len(avail_idxs)}] samples belonging to the desired datasets from '
                      f'[{len(self.avail_idxs)}] original samples. ({len(avail_idxs) / len(self.avail_idxs) * 100:.2f}%)')
                self.avail_idxs = avail_idxs
        if items is None and prefix == 'train' and self.hparams.get('dataset_downsample_rate', 1.0) < 1.0 \
                and self.avail_idxs is not None:
            ratio = self.hparams.get('dataset_downsample_rate', 1.0)
            orig_len = len(self.avail_idxs)
            tgt_len = round(orig_len * ratio)
            self.avail_idxs = np.random.choice(self.avail_idxs, size=tgt_len, replace=False).tolist()
            print(f'| Downsamping training set with ratio [{ratio * 100:.2f}%], '
                  f'[{tgt_len}] samples of [{orig_len}] samples are selected.')
        if items is None:
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def add_noise(self, clean_wav):
        if self.noise_ds is None:   # each instance in multiprocessing must create unique ds object
            self.noise_ds = IndexedDataset(f"{self.hparams['noise_data_dir']}/{self.prefix}")
        noise_idx = np.random.randint(len(self.noise_ds))
        noise_item = self.noise_ds[noise_idx]
        noise_wav = noise_item['feat']

        if type(self.noise_snr) == tuple:
            snr = np.random.rand() * (self.noise_snr[1] - self.noise_snr[0]) + self.noise_snr[0]
        else:
            snr = self.noise_snr
        clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
        if len(clean_wav) > len(noise_wav):
            ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
            noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
        if len(clean_wav) < len(noise_wav):
            start = 0
            noise_wav = noise_wav[start: start + len(clean_wav)]
        noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1)) + 1e-5
        adjusted_noise_rms = clean_rms / (10 ** (snr / 20) + 1e-5)
        adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
        mixed = clean_wav + adjusted_noise_wav
        # Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
            if mixed.max(axis=0) >= abs(mixed.min(axis=0)):
                reduction_rate = max_int16 / mixed.max(axis=0)
            else:
                reduction_rate = min_int16 / mixed.min(axis=0)
            mixed = mixed * reduction_rate
        return mixed

    @staticmethod
    def process_mel(wav_fn, hparams):
        wav2spec_dict = librosa_wav2spec(
            wav_fn,
            fft_size=hparams['fft_size'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'])
        mel = wav2spec_dict['mel']
        del wav2spec_dict['wav'], wav2spec_dict['linear'], wav2spec_dict['mel_basis'], wav2spec_dict['wav_orig']
        return mel

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)

        wav = item['wav']
        noise_added = np.random.rand() < hparams.get('noise_prob', 0.8)
        if self.prefix == 'test' and not hparams.get('noise_in_test', False):
            noise_added = False
        if noise_added:
            wav = self.add_noise(wav)
            # print('noise added!')
        mel = TEDataset.process_mel(wav, self.hparams)
        assert len(mel) == self.sizes[index], (len(mel), self.sizes[index])
        max_frames = hparams['max_frames']
        mel2ph_len = sum((item["mel2ph"] > 0).astype(int))
        T = min(item['len'], mel2ph_len, len(item['f0']))
        real_len = T
        T = math.ceil(min(T, max_frames) / hparams['frames_multiple']) * hparams['frames_multiple']
        spec = torch.Tensor(mel)[:max_frames]
        spec = pad_or_cut_xd(spec, T, dim=0)
        if 5 < hparams.get('use_mel_bins', hparams['audio_num_mel_bins']) < hparams['audio_num_mel_bins']:
            spec = spec[:, :hparams.get('use_mel_bins', 80)]
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        if hparams['use_spk_embed']:
            sample["spk_embed"] = torch.Tensor(item['spk_embed'])
        if hparams['use_spk_id']:
            sample["spk_id"] = int(item['spk_id'])
        if noise_added and self.prefix in ['train', 'valid']:
            noisy_mel = add_plain_noise(sample['mel'], type_and_std=hparams.get('mel_add_noise', 'none'))
            sample['mel'] = torch.clamp(noisy_mel, hparams['mel_vmin'], hparams['mel_vmax'])
        sample['mel2ph'] = mel2ph = pad_or_cut_xd(torch.LongTensor(item['mel2ph']), T, 0)
        sample["mel2word"] = mel2word = pad_or_cut_xd(torch.LongTensor(item.get("mel2word")), T, 0)
        sample['mel_nonpadding'] = pad_or_cut_xd(sample['mel_nonpadding'].float(), T, 0)
        ph_bd = torch.zeros_like(mel2ph)
        for idx in range(1, T):
            if mel2ph[idx] == 0:
                break
            if mel2ph[idx] != mel2ph[idx-1]:
                ph_bd[idx] = 1
        sample['ph_bd'] = ph_bd.long()

        if hparams['use_pitch_embed']:
            assert 'f0' in item
            # pitch = torch.LongTensor(item.get(hparams.get('pitch_key', 'pitch')))[:T]
            f0, uv = norm_interp_f0(item["f0"][:T])
            # 给 f0 加噪
            if noise_added and self.prefix in ['train', 'valid']:
                f0 = torch.FloatTensor(f0)
                f0[uv == 0] = add_plain_noise(f0[uv == 0], type_and_std=hparams.get('f0_add_noise', 'none'))
            uv = pad_or_cut_xd(torch.FloatTensor(uv), T, 0)
            f0 = pad_or_cut_xd(torch.FloatTensor(f0), T, 0)
            pitch_coarse = f0_to_coarse(denorm_f0(f0, uv))
        else:
            f0, uv, pitch, pitch_coarse = None, None, None, None
        sample["f0"], sample["uv"], sample["pitch_coarse"] = f0, uv, pitch_coarse

        if hparams.get('use_breathiness', False):
            breathiness = torch.FloatTensor(item['breathiness'])
            if noise_added and self.prefix in ['train', 'valid']:
                breathiness = add_plain_noise(breathiness, type_and_std=hparams.get('breathiness_add_noise', 'none'), x_min=0, x_max=0.8)
            breathiness_coarse = energy_to_coarse(breathiness)
            breathiness_coarse = pad_or_cut_xd(breathiness_coarse, T, 0)
            sample['breathiness'] = breathiness_coarse

        if hparams.get('use_energy', False):
            energy = torch.FloatTensor(item['energy'])
            if noise_added and self.prefix in ['train', 'valid']:
                energy = add_plain_noise(energy, type_and_std=hparams.get('energy_add_noise', 'none'), x_min=0, x_max=0.8)
            energy_coarse = energy_to_coarse(energy)
            energy_coarse = pad_or_cut_xd(energy_coarse, T, 0)
            sample['energy'] = energy_coarse

        if hparams.get('use_zcr', False):
            zcr = torch.FloatTensor(item['zcr'])
            if noise_added and self.prefix in ['train', 'valid']:
                zcr = add_plain_noise(zcr, type_and_std=hparams.get('zcr_add_noise', 'none'), x_min=0, x_max=1)
            zcr_coarse = anything_to_coarse(zcr, bins=256, x_max=1.0, x_min=0.0, pad=0)
            zcr_coarse = pad_or_cut_xd(zcr_coarse, T, 0)
            sample['zcr'] = zcr_coarse

        # make tech matrix
        mix_tech = torch.LongTensor(item['mix_tech'])
        falsetto_tech = torch.LongTensor(item['falsetto_tech'])
        breathy_tech = torch.LongTensor(item['breathy_tech'])
        pharyngeal_tech = torch.LongTensor(item['pharyngeal_tech'])
        vibrato_tech = torch.LongTensor(item['vibra_tech'])
        glissando_tech = torch.LongTensor(item['glissando_tech'])
        tech_matrix = torch.stack((mix_tech, falsetto_tech, breathy_tech, pharyngeal_tech, vibrato_tech, glissando_tech), dim=1)
        sample['tech'] = tech_matrix    # [T, C]

        # delete big redundancy
        if not hparams.get('use_mel', True) and 'mel' in sample:
            sample['mel'] = None
            if 'mel_nonpadding' in sample:
                sample['mel_nonpadding'] = None
        if not hparams.get('use_wav', False) and 'wav' in sample:
            sample['wav'] = None

        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        # text = [s['text'] for s in samples]
        mels = collate_1d_or_2d([s['mel'] for s in samples], 0.0) if 'mel' in samples[0] else None
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples]) if 'mel' in samples[0] else 0
        mel_nonpadding = collate_1d_or_2d([s['mel_nonpadding'] for s in samples], 0.0) if 'mel_nonpadding' in samples[0] else None

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'mels': mels,
            'mel_lengths': mel_lengths,
            'mel_nonpadding': mel_nonpadding
        }

        if hparams['use_spk_embed']:
            spk_embed = torch.stack([s['spk_embed'] for s in samples])
            batch['spk_embed'] = spk_embed
        if hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids

        if hparams['use_pitch_embed']:
            f0 = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            # pitch = collate_1d_or_2d([s['pitch'] for s in samples])
            uv = collate_1d_or_2d([s['uv'] for s in samples])
            pitch_coarse = collate_1d_or_2d([s['pitch_coarse'] for s in samples])
        else:
            f0, uv, pitch, pitch_coarse = None, None, None, None
        batch['mel2ph'] = collate_1d_or_2d([s['mel2ph'] for s in samples], 0.0)
        batch['f0'], batch['uv'], batch['pitch_coarse'] = f0, uv, pitch_coarse
        batch["wav"] = collate_1d_or_2d([s['wav'] for s in samples], 0.0) if 'wav' in samples[0] else None

        batch["ph_bd"] = collate_1d_or_2d([s['ph_bd'] for s in samples], 0.0)
        batch['mel2word'] = collate_1d_or_2d([s['mel2word'] for s in samples], 0)
        batch['techs'] = collate_1d_or_2d([s['tech'] for s in samples], 0)

        batch['breathiness'] = collate_1d_or_2d([s['breathiness'] for s in samples], 0) if 'breathiness' in samples[0] else None
        batch['energy'] = collate_1d_or_2d([s['energy'] for s in samples], 0) if 'energy' in samples[0] else None
        batch['zcr'] = collate_1d_or_2d([s['zcr'] for s in samples], 0) if 'zcr' in samples[0] else None

        return batch

    @property
    def num_workers(self):
        if self.prefix == 'train':
            return int(os.getenv('NUM_WORKERS', self.hparams['ds_workers']))
        return 1

# TODO:
def make_mix_batch(ph_lst, mel2ph_lst, seg_keys=('breathe', '_NONE')):
    # 这个没有对 spk 进行规约
    src_bsz = len(ph_lst)

    seg_pool = []
    seg_idxs = []
    for item_idx in range(src_bsz):
        ph = ph_lst[item_idx]
        mel2ph = mel2ph_lst[item_idx]
        last_mel2ph_idx = 0
        last_ph_idx = 0
        for ph_idx in range(1, len(ph)-1):
            p = ph[ph_idx]
            if p in seg_keys:
                for mel2ph_idx in range(last_mel2ph_idx, len(mel2ph)):
                    if mel2ph[mel2ph_idx] - 1 == ph_idx:
                        seg_pool.append([item_idx, last_mel2ph_idx, mel2ph_idx, last_ph_idx, ph_idx, mel2ph_idx - last_mel2ph_idx])
                        last_mel2ph_idx = mel2ph_idx
                        break
                last_ph_idx = ph_idx
        seg_pool.append([item_idx, last_mel2ph_idx, len(mel2ph), last_ph_idx, len(ph), len(mel2ph) - last_mel2ph_idx])


class TEMixDataset(TEDataset):
    """
    对 batch 进行 intra-batch mixing
    intra-batch mixing 貌似就不能使用 musan noise 了，不过反正都可以试试
    """
    def __getitem__(self, index):
        sample = super(TEMixDataset, self).__getitem__(index)
        item = self._get_item(index)
        sample['ph'] = item['ph']

    def collater(self, samples):
        if self.prefix != 'train':
            return super(TEMixDataset, self).collater(samples)
        else:
            if len(samples) == 0:
                return {}
            hparams = self.hparams
            id = torch.LongTensor([s['id'] for s in samples])
            item_names = [s['item_name'] for s in samples]

            batch = {}

            ph_lst = [s['ph'] for s in samples]
            mel2ph_lst = [s['mel2ph'] for s in samples]


            mels = collate_1d_or_2d([s['mel'] for s in samples], 0.0) if 'mel' in samples[0] else None
            mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples]) if 'mel' in samples[0] else 0
            mel_nonpadding = collate_1d_or_2d([s['mel_nonpadding'] for s in samples], 0.0) if 'mel_nonpadding' in samples[0] else None

            batch = {
                'id': id,
                'item_name': item_names,
                'nsamples': len(samples),
                'mels': mels,
                'mel_lengths': mel_lengths,
                'mel_nonpadding': mel_nonpadding
            }

            if hparams['use_spk_embed']:
                spk_embed = torch.stack([s['spk_embed'] for s in samples])
                batch['spk_embed'] = spk_embed
            if hparams['use_spk_id']:
                spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
                batch['spk_ids'] = spk_ids

            if hparams['use_pitch_embed']:
                f0 = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
                # pitch = collate_1d_or_2d([s['pitch'] for s in samples])
                uv = collate_1d_or_2d([s['uv'] for s in samples])
                pitch_coarse = collate_1d_or_2d([s['pitch_coarse'] for s in samples])
            else:
                f0, uv, pitch, pitch_coarse = None, None, None, None
            batch['mel2ph'] = collate_1d_or_2d([s['mel2ph'] for s in samples], 0.0)
            batch['f0'], batch['uv'], batch['pitch_coarse'] = f0, uv, pitch_coarse
            batch["wav"] = collate_1d_or_2d([s['wav'] for s in samples], 0.0) if 'wav' in samples[0] else None

            batch["ph_bd"] = collate_1d_or_2d([s['ph_bd'] for s in samples], 0.0)
            batch['mel2word'] = collate_1d_or_2d([s['mel2word'] for s in samples], 0)
            batch['techs'] = collate_1d_or_2d([s['tech'] for s in samples], 0)

            batch['breathiness'] = collate_1d_or_2d([s['breathiness'] for s in samples], 0)
            batch['energy'] = collate_1d_or_2d([s['energy'] for s in samples], 0)
            batch['zcr'] = collate_1d_or_2d([s['zcr'] for s in samples], 0)

            return batch

