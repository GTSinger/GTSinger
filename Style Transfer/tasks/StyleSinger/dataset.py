import matplotlib
matplotlib.use('Agg')
from utils.pitch_utils import norm_interp_f0, denorm_f0
from utils.hparams import hparams
import torch
import torch.optim
import torch.nn.functional as F
import torch.utils.data
from utils.indexed_datasets import IndexedDataset
import torch.distributions
import numpy as np
from utils.commons.dataset_utils import BaseDataset, collate_1d_or_2d
from utils.commons.indexed_datasets import IndexedDataset
    

class BaseSpeechDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(shuffle)
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
            if prefix == 'test' and len(hparams['test_ids']) > 0:
                self.avail_idxs = hparams['test_ids']
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == 'train' and hparams['min_frames'] > 0:
                self.avail_idxs = [x for x in self.avail_idxs if self.sizes[x] >= hparams['min_frames']]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        assert len(item['mel']) == self.sizes[index], (len(item['mel']), self.sizes[index])
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item['mel'])[:max_frames]
        max_frames = spec.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
        spec = spec[:max_frames]
        ph_token = torch.LongTensor(item['ph_token'][:hparams['max_input_tokens']])
        sample = {
            "id": index,
            "item_name": item['item_name'],
            # "text": item['txt'],
            "txt_token": ph_token,
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        if hparams['use_spk_embed']:
            sample["spk_embed"] = torch.Tensor(item['spk_embed'])
        if hparams['use_spk_id']:
            sample["spk_id"] = int(item['spk_id'])
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        # text = [s['text'] for s in samples]
        txt_tokens = collate_1d_or_2d([s['txt_token'] for s in samples], 0)
        mels = collate_1d_or_2d([s['mel'] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            # 'text': text,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'mels': mels,
            'mel_lengths': mel_lengths,
        }

        if hparams['use_spk_embed']:
            spk_embed = torch.stack([s['spk_embed'] for s in samples])
            batch['spk_embed'] = spk_embed
        if hparams['use_spk_id']:
            spk_id = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_id'] = spk_id
        return batch
    

class BaseSingerdataset(BaseSpeechDataset):
    def __getitem__(self, index):
        sample = super(BaseSingerdataset, self).__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        mel = sample['mel']
        mel2ph_len = sum((item["mel2ph"] > 0).astype(np.int))
        T = min(mel.shape[0], mel2ph_len, len(item["f0"]))
        sample['mel'] = mel[:T]
        ph_token = sample['txt_token']
        sample['mel2ph'] = mel2ph = torch.LongTensor(item['mel2ph'])[:T]
        if hparams['use_pitch_embed']:
            assert 'f0' in item
            # pitch = torch.LongTensor(item.get(hparams.get('pitch_key', 'pitch')))[:T]
            f0, uv = norm_interp_f0(item["f0"][:T], hparams)
            uv = torch.FloatTensor(uv)
            f0 = torch.FloatTensor(f0)
            if hparams['pitch_type'] == 'ph':
                if "f0_ph" in item:
                    f0 = torch.FloatTensor(item['f0_ph'])
                else:
                    f0 = denorm_f0(f0, None)
                f0_phlevel_sum = torch.zeros_like(ph_token).float().scatter_add(0, mel2ph - 1, f0)
                f0_phlevel_num = torch.zeros_like(ph_token).float().scatter_add(
                    0, mel2ph - 1, torch.ones_like(f0)).clamp_min(1)
                f0_ph = f0_phlevel_sum / f0_phlevel_num
                f0, uv = norm_interp_f0(f0_ph)
        else:
            f0, uv, pitch = None, None, None
        # sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch
        sample["f0"], sample["uv"] = f0, uv
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(BaseSingerdataset, self).collater(samples)
        hparams = self.hparams
        if hparams['use_pitch_embed']:
            f0 = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            # pitch = collate_1d_or_2d([s['pitch'] for s in samples])
            uv = collate_1d_or_2d([s['uv'] for s in samples])
        else:
            f0, uv, pitch = None, None, None
        mel2ph = collate_1d_or_2d([s['mel2ph'] for s in samples], 0.0)
        batch.update({
            'mel2ph': mel2ph,
            # 'pitch': pitch,
            'f0': f0,
            'uv': uv,
        })
        return batch
    

class StyleSinger_dataset(BaseSingerdataset):
    def __getitem__(self, index):
        hparams=self.hparams
        sample = super(StyleSinger_dataset, self).__getitem__(index)
        item = self._get_item(index)
        max_frames = sample['mel'].shape[0]
        note = torch.LongTensor(item['ep_pitches'][:hparams['max_input_tokens']])
        note_dur = torch.FloatTensor(item['ep_notedurs'][:hparams['max_input_tokens']])
        note_type = torch.LongTensor(item['ep_types'][:hparams['max_input_tokens']])
        sample["note"], sample["note_dur"], sample["note_type"] = note, note_dur, note_type
        if hparams['emo']:
            sample["emo_embed"] = torch.Tensor(item['emo_embed'])
        sample["spk_embed"] = torch.Tensor(item['spk_embed'])

        return sample
    
    def collater(self, samples):
        hparams=self.hparams
        if len(samples) == 0:
            return {}
        batch = super(StyleSinger_dataset, self).collater(samples)
        notes = collate_1d_or_2d([s['note'] for s in samples], 0.0)
        note_durs = collate_1d_or_2d([s['note_dur'] for s in samples], 0.0)
        note_types = collate_1d_or_2d([s['note_type'] for s in samples], 0.0)
        batch["notes"], batch["note_durs"], batch["note_types"] = notes, note_durs, note_types
        spk_embed = torch.stack([s['spk_embed'] for s in samples])
        batch['spk_embed'] = spk_embed
        if hparams['emo']:
            emo_embed = torch.stack([s['emo_embed'] for s in samples])
            batch['emo_embed'] = emo_embed
            
        return batch