import os
import sys
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing.pool import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import traceback
from scipy.ndimage import gaussian_filter

import utils
from utils.hparams import hparams
from data_gen.tts.data_gen_utils import get_pitch
from utils.indexed_datasets import IndexedDataset
from utils.pitch_utils import norm_interp_f0, denorm_f0, midi_pitch_shift, midi_to_hz, get_uv
import utils.audio as audio
from utils.plot import spec_to_figure
from utils.text_encoder import TokenTextEncoder
from tasks.base_task import BaseDataset
from tasks.tts.fs2 import FastSpeech2Task
from tasks.tts.transformer_tts import TransformerTtsTask
from vocoders.base_vocoder import get_vocoder_cls, BaseVocoder

from modules.speech2singing.diffvae import DiffVAE
from modules.speech2singing.diffvae2 import DiffVAE2, DiffVAE3

from modules.diffusion.net import DiffNet
from modules.fastspeech.pe import PitchExtractor
from modules.fastspeech.tts_modules import mel2ph_to_dur

MODELS = {
    'diffvae1': DiffVAE,
    'diffvae2': DiffVAE2,
    'diffvae3': DiffVAE3
}

class Speech2SingingDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, encoder=None):
        super(Speech2SingingDataset, self).__init__(shuffle)
        self.data_dir = hparams['binary_data_dir']
        self.prefix = prefix
        self.hparams = hparams
        self.speech_sizes = np.load(f'{self.data_dir}/{self.prefix}_speech_lengths.npy')
        self.sing_sizes = np.load(f'{self.data_dir}/{self.prefix}_sing_lengths.npy')
        self.sizes = self.speech_sizes  # NOTE for default
        self.indexed_ds = None
        self.encoder = encoder
        self.pad_idx = encoder.pad()
        self.eos_idx = encoder.eos()

        # pitch stats
        speech_f0_stats_fn = f'{self.data_dir}/train_speech_f0s_mean_std.npy'
        sing_f0_stats_fn = f'{self.data_dir}/train_sing_f0s_mean_std.npy'
        if os.path.exists(speech_f0_stats_fn):
            hparams['speech_f0_mean'], hparams['speech_f0_std'] = self.speech_f0_mean, self.speech_f0_std = np.load(speech_f0_stats_fn)
            hparams['speech_f0_mean'] = float(hparams['speech_f0_mean'])
            hparams['speech_f0_std'] = float(hparams['speech_f0_std'])
        else:
            hparams['speech_f0_mean'], hparams['speech_f0_std'] = self.speech_f0_mean, self.speech_f0_std = None, None
        if os.path.exists(sing_f0_stats_fn):
            hparams['sing_f0_mean'], hparams['sing_f0_std'] = self.sing_f0_mean, self.sing_f0_std = np.load(sing_f0_stats_fn)
            hparams['sing_f0_mean'] = float(hparams['sing_f0_mean'])
            hparams['sing_f0_std'] = float(hparams['sing_f0_std'])

        else:
            hparams['sing_f0_mean'], hparams['sing_f0_std'] = self.sing_f0_mean, self.sing_f0_std = None, None

        if prefix == 'test':
            if hparams['test_input_dir'] != '':
                self.indexed_ds, self.sizes = self.load_test_inputs(hparams['test_input_dir'])
            else:
                if hparams['num_test_samples'] > 0:
                    self.avail_idxs = list(range(hparams['num_test_samples'])) + hparams['test_ids']
                    self.speech_sizes = np.array([self.speech_sizes[i] for i in self.avail_idxs], dtype=int)
                    self.sing_sizes = np.array([self.sing_sizes[i] for i in self.avail_idxs], dtype=int)

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        max_frames = hparams['max_frames']
        speech_mel = torch.Tensor(item['speech']['mel'])[:max_frames] if hparams.get('use_speech_mel', False) else None
        speech_spenv = torch.Tensor(item['speech']['spenv'])[:max_frames] if hparams.get('use_speech_spenv', False) else None
        speech_mel_mono = torch.Tensor(item['speech']['spmel_mono'])[:max_frames] if hparams.get('use_speech_mel_mono', False) else None
        speech_wav2vec2_mono = torch.Tensor(item['speech']['wav2vec2_mono'])[:max_frames]\
            if hparams.get('use_speech_wav2vec2_mono', False) else None
        sing_mel = torch.Tensor(item['sing']['mel'])[:max_frames]
        speech_energy = (speech_mel.exp() ** 2).sum(-1).sqrt() if speech_mel is not None else None
        sing_energy = (sing_mel.exp() ** 2).sum(-1).sqrt()
        sing_spenv = torch.Tensor(item['sing']['spenv'])[:max_frames] if hparams.get('use_sing_spenv', False) else None
        sing_wav2vec2 = torch.Tensor(item['sing']['wav2vec2'])[:max_frames]\
            if hparams.get('use_sing_wav2vec2', False) else None
        sing_spenv_reduced = None
        if hparams.get('use_sing_spenv_reduced', False):
            if sing_spenv is not None:
                sing_spenv_reduced = (sing_spenv.exp() ** 2).sum(-1).sqrt()
                sing_spenv_reduced = torch.sigmoid((sing_spenv_reduced - sing_spenv_reduced.mean())
                                                   / (sing_spenv_reduced.std() + 1e-8) * hparams.get('sing_spenv_reduced_coef', 1.0))
            else:
                sing_spenv_reduced = torch.sigmoid((sing_energy - sing_energy.mean())
                                                   / (sing_energy.std() + 1e-8) * hparams.get('sing_spenv_reduced_coef', 1.0))
            if hparams.get('spenv_filter', 0) > 0:
                sing_spenv_reduced = torch.Tensor(gaussian_filter(sing_spenv_reduced, sigma=hparams.get('spenv_filter', 0)))
        if hparams.get('use_speech_spenv_reduced', False):
            speech_energy_mono = (speech_mel_mono.exp() ** 2).sum(-1).sqrt()
            speech_energy_mono = torch.sigmoid((speech_energy_mono - speech_energy_mono.mean())
                                               / (speech_energy_mono.std() + 1e-8) * hparams.get('sing_spenv_reduced_coef', 1.0))
        speech_mel2ph = torch.LongTensor(item['speech']['mel2ph'])[:max_frames] if 'mel2ph' in item['speech'] else None
        sing_mel2ph = torch.LongTensor(item['sing']['mel2ph'])[:max_frames] if 'mel2ph' in item['sing'] else None
        speech_f0 = item['speech']["f0"][:max_frames]
        speech_uv = get_uv(speech_f0)
        sing_f0 = item['sing']["f0"][:max_frames]
        sing_uv = get_uv(sing_f0)
        speech_uv = torch.FloatTensor(speech_uv)
        sing_uv = torch.FloatTensor(sing_uv)
        speech_f0 = torch.FloatTensor(speech_f0)
        sing_f0 = torch.FloatTensor(sing_f0)
        phone = torch.LongTensor(item['phone'][:hparams['max_input_tokens']]) if 'phone' in item else None
        speech_pitch = torch.LongTensor(item['speech'].get("pitch"))[:max_frames] if "pitch" in item['speech'] else None
        sing_pitch = torch.LongTensor(item['sing'].get("pitch"))[:max_frames] if "pitch" in item['sing'] else None

        sample = {
            "id": index,
            "item_name": item['item_name'],
            "text": item['txt'],
            "txt_token": phone,
            "speech": {
                "mel": speech_mel if hparams['use_speech_mel'] else None,
                "spenv": speech_spenv if hparams['use_speech_spenv'] else None,
                "spenv_reduced": speech_energy_mono if hparams.get('use_speech_spenv_reduced', False) else None,
                "mel_mono": speech_mel_mono if hparams.get('use_speech_mel_mono', True) else None,
                "wav2vec2_mono": speech_wav2vec2_mono,
                "pitch": speech_pitch,
                "energy": speech_energy,
                "f0": speech_f0,
                "uv": speech_uv,
                "mel2ph": speech_mel2ph,
                "mel_nonpadding": speech_mel.abs().sum(-1) > 0 if hparams['use_speech_mel'] else None,
                "dur": item['speech']['dur'] if 'dur' in item['speech'] else None
            },
            "sing": {
                "mel": sing_mel,
                "spenv": sing_spenv,
                "spenv_reduced": sing_spenv_reduced,
                "wav2vec2": sing_wav2vec2,
                "pitch": sing_pitch,
                "energy": sing_energy,
                "f0": sing_f0,
                # "denorm_f0": sing_denorm_f0,
                "uv": sing_uv,
                "mel2ph": sing_mel2ph,
                "mel_nonpadding": sing_mel.abs().sum(-1) > 0,
                "dur": item['sing']['dur'] if 'dur' in item['sing'] else None
            }
        }
        if self.hparams['use_spk_embed']:
            sample["spk_embed"] = torch.Tensor(item['spk_embed'])
        if 'spk_id' in item:
            sample["spk_id"] = item['spk_id']

        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        ids = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        text = [s['text'] for s in samples]
        txt_tokens = utils.collate_1d([s['txt_token'] for s in samples], 0) \
            if samples[0]['txt_token'] is not None else None
        prev_tokens = utils.collate_1d([s['txt_token'] for s in samples], 0, shift_right=True) \
            if samples[0]['txt_token'] is not None else None
        speech_f0 = utils.collate_1d([s['speech']['f0'] for s in samples], 0.0)
        sing_f0 = utils.collate_1d([s['sing']['f0'] for s in samples], 0.0)
        # sing_denorm_f0 = utils.collate_1d([s['sing']['denorm_f0'] for s in samples], 0.0)
        speech_pitch = utils.collate_1d([s['speech']['pitch'] for s in samples])\
            if samples[0]['speech']['pitch'] is not None else None
        sing_pitch = utils.collate_1d([s['sing']['pitch'] for s in samples])\
            if samples[0]['sing']['pitch'] is not None else None
        speech_uv = utils.collate_1d([s['speech']['uv'] for s in samples])
        sing_uv = utils.collate_1d([s['sing']['uv'] for s in samples])
        speech_energy = utils.collate_1d([s['speech']['energy'] for s in samples], 0.0) \
            if samples[0]['speech']['energy'] is not None else None
        sing_energy = utils.collate_1d([s['sing']['energy'] for s in samples], 0.0)
        speech_mel2ph = utils.collate_1d([s['speech']['mel2ph'] for s in samples], 0.0) \
            if samples[0]['speech']['mel2ph'] is not None else None
        sing_mel2ph = utils.collate_1d([s['sing']['mel2ph'] for s in samples], 0.0) \
            if samples[0]['sing']['mel2ph'] is not None else None
        speech_mels = utils.collate_2d([s['speech']['mel'] for s in samples], 0.0) \
            if samples[0]['speech']['mel'] is not None else None
        speech_spenvs = utils.collate_2d([s['speech']['spenv'] for s in samples], 0.0) \
            if samples[0]['speech']['spenv'] is not None else None
        speech_spenvs_reduced = utils.collate_1d([s['speech']['spenv_reduced'] for s in samples], 0.0)[:, :, None] \
            if samples[0]['speech']['spenv_reduced'] is not None else None  # [B, T] -> [B, T, 1]
        speech_mels_mono = utils.collate_2d([s['speech']['mel_mono'] for s in samples], 0.0) \
            if samples[0]['speech']['mel_mono'] is not None else None
        speech_wav2vec2s_mono = utils.collate_2d([s['speech']['wav2vec2_mono'] for s in samples], 0.0) \
            if samples[0]['speech']['wav2vec2_mono'] is not None else None
        sing_mels = utils.collate_2d([s['sing']['mel'] for s in samples], 0.0)
        sing_spenvs = utils.collate_2d([s['sing']['spenv'] for s in samples], 0.0) \
            if samples[0]['sing']['spenv'] is not None else None
        sing_spenvs_reduced = utils.collate_1d([s['sing']['spenv_reduced'] for s in samples], 0.0)[:, :, None] \
            if samples[0]['sing']['spenv_reduced'] is not None else None    # [B, T] -> [B, T, 1]
        sing_wav2vec2s = utils.collate_2d([s['sing']['wav2vec2'] for s in samples], 0.0) \
            if samples[0]['sing']['wav2vec2'] is not None else None
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples]) \
            if samples[0]['txt_token'] is not None else None
        speech_mel_lengths = torch.LongTensor([s['speech']['mel'].shape[0] for s in samples]) \
            if hparams['use_speech_mel'] else None
        sing_mel_lengths = torch.LongTensor([s['sing']['mel'].shape[0] for s in samples])

        batch = {
            'id': ids,
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'txt_tokens': txt_tokens,
            'prev_tokens': prev_tokens,
            'txt_lengths': txt_lengths,
            'speech': {
                'mels': speech_mels,
                'spenvs': speech_spenvs,
                'spenvs_reduced': speech_spenvs_reduced,
                'mels_mono': speech_mels_mono,
                'wav2vec2s_mono': speech_wav2vec2s_mono,
                'mel_lengths': speech_mel_lengths,
                'mel2ph': speech_mel2ph,
                'energy': speech_energy,
                'pitch': speech_pitch,
                'f0': speech_f0,
                'uv': speech_uv
            },
            'sing': {
                'mels': sing_mels,
                'spenvs': sing_spenvs,
                'spenvs_reduced': sing_spenvs_reduced,
                'wav2vec2s': sing_wav2vec2s,
                'mel_lengths': sing_mel_lengths,
                'mel2ph': sing_mel2ph,
                'energy': sing_energy,
                'pitch': sing_pitch,
                'f0': sing_f0,
                # 'denorm_f0': sing_denorm_f0,
                'uv': sing_uv
            }
        }

        if self.hparams['use_spk_embed']:
            spk_embed = torch.stack([s['spk_embed'] for s in samples])
            batch['spk_embed'] = spk_embed
        if 'spk_id' in samples[0]:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids

        return batch

    def num_tokens(self, index, song_type="speech"):
        return self.size(index, song_type)

    def size(self, index, song_type="speech"):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        # size = min(self._sizes(index, song_type), hparams['max_frames'])
        # return size
        if song_type == "":
            sizes = self.speech_sizes + self.sing_sizes
        elif song_type == "avg":
            sizes = (self.speech_sizes + self.sing_sizes) / 2
        elif song_type == "speech":
            sizes = self.sizes
        elif song_type == "sing":
            sizes = self.sing_sizes
        return min(sizes[index], hparams['max_frames'])


class DiffVAETask(FastSpeech2Task):
    def __init__(self):
        super(DiffVAETask, self).__init__()
        self.dataset_cls = partial(Speech2SingingDataset, encoder=self.phone_encoder)
        self.pe = None
        if hparams.get('use_asr', False) and hparams.get('use_bpe', False):
            self.token_encoder = self.build_token_encoder(hparams['binary_data_dir'])
        else:
            self.token_encoder = self.phone_encoder

    def build_tts_model(self):
        self.model = MODELS[hparams.get('diffvae_type', 'diffvae2')](
            self.phone_encoder, partial(DiffNet, hparams['audio_num_mel_bins']))
        if hparams.get('model_ckpt', ''):
            utils.load_ckpt(self.model, hparams['model_ckpt'], 'model', strict=True)
        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            self.pe = PitchExtractor()
            # self.pe.cuda()
            utils.load_ckpt(self.pe, hparams['pe_ckpt'], 'model', strict=True)
            self.pe.eval()
        if hparams.get('asr_ckpt', '') != '' and hparams.get('use_asr', False):
            utils.load_ckpt(self.model.asr, hparams['asr_ckpt'], 'model', strict=True)
            # self.model.fs2.decoder = None
            if hparams.get('freeze_asr', True):
                for k, v in self.model.asr.named_parameters():
                    v.requires_grad = False

    def _training_step(self, sample, batch_idx, _):
        log_outputs = self.run_model(self.model, sample)
        total_loss = sum([v for v in log_outputs.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        log_outputs['batch_size'] = sample['id'].size()[0]
        log_outputs['lr'] = self.scheduler.get_lr()
        return total_loss, log_outputs

    def run_model(self, model, sample, return_output=False, infer=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        prev_tokens = sample['prev_tokens']
        sing_mels = sample['sing']['mels']  # [B, T_s, 80]
        sing_spenvs = sample['sing']['spenvs'] if hparams.get('use_sing_spenv', False) else None
        if hparams.get('use_sing_spenv_reduced', False):
            sing_spenvs = sample['sing']['spenvs_reduced']
        speech_mels = sample['speech']['mels']
        speech_spenvs = sample['speech']['spenvs']
        if hparams.get('use_sing_spenv_reduced', False):
            speech_spenvs = sample['speech']['spenvs_reduced']
        speech_mels_mono = sample['speech']['mels_mono']
        source_mels = speech_mels_mono
        if (not hparams.get("use_speech_mel_mono", True)) and hparams.get("use_speech_mel", False):
            source_mels = speech_mels
        if hparams.get("use_speech_wav2vec2_mono", False):
            speech_wav2vec2s_mono = sample['speech']['wav2vec2s_mono']
            source_mels = speech_wav2vec2s_mono
        sing_wav2vec2s = sample['sing']['wav2vec2s'] if hparams.get("use_sing_wav2vec2", False) else None
        if hparams.get("use_sing_input", False):
            source_mels = sing_wav2vec2s if hparams.get("use_sing_wav2vec2", False) else sing_mels
        # mel2ph = sample['mel2ph'] if hparams['use_gt_dur'] else None # [B, T_s]
        mel2ph = sample['sing']['mel2ph']
        f0 = sample['sing']['f0']
        if hparams.get('debug', False):
            print('run_model f0----------run_model f0---------')
            print(f0)
        # denorm_f0 = sample['sing']['denorm_f0']
        uv = sample['sing']['uv']
        energy = sample['sing']['energy']
        # fs2_mel = sample['fs2_mels']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        if hparams['pitch_type'] == 'cwt':
            cwt_spec = sample[f'cwt_spec']
            f0_mean = sample['f0_mean']
            f0_std = sample['f0_std']
            sample['f0_cwt'] = f0 = model.cwt2f0_norm(cwt_spec, f0_mean, f0_std, mel2ph)

        output = model(source_mels, f0, speech_spenvs, spk_embed, ref_mels=sing_mels, ref_spenv=sing_spenvs,
                       txt_tokens=txt_tokens, mel2ph=mel2ph, uv=uv, energy=energy, infer=infer,
                       forcing=self.global_step < hparams.get('forcing', 0), prev_tokens=prev_tokens)

        losses = {}
        if hparams.get('diffvae_decoder', 'diffsinger') in ['diffsinger', 'ddim'] and 'diff_loss' in output:
            losses['mel'] = output['diff_loss']
        elif hparams.get('diffvae_decoder', 'diffsinger') in ['fs2', 'prodiff']:
            self.add_mel_loss(output['mel_out'], sing_mels, losses, postfix='_mel')
        if 'dur' in output:
            self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        if hparams['use_pitch_embed'] and 'pitch_pred' in output:
            self.add_pitch_loss(output, sample, losses)
        if hparams['use_energy_embed'] and 'energy_pred' in output:
            self.add_energy_loss(output['energy_pred'], energy, losses)
        if hparams['use_sing_spenv'] and 'spenv_pred' in output:
            self.add_mel_loss(output['spenv_pred'], sing_spenvs, losses, postfix='_spenv')
        if (hparams['use_sing_spenv'] or hparams.get('use_sing_spenv_reduced', False)) \
                and 'rq_loss' in output and 'r_loss' in output:
            self.add_vqvae_loss(output, losses)
        # add attn guided loss
        if self.global_step > hparams.get('forcing', 0):
            gloss = 0
            for k in output.keys():
                if k[:6] == 'gloss_':
                    gloss += output[k]
            losses['gloss'] = gloss * hparams.get('lambda_guided_loss', 0.5)
            # losses['gloss'] = (output['gloss_f0'] + output['gloss_spenv']) * hparams.get('lambda_guided_loss', 0.5)
        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample, batch_idx):
        outputs = {}

        txt_tokens = sample['txt_tokens']  # [B, T_t]
        prev_tokens = sample['prev_tokens']
        speech_spenvs = sample['speech']['spenvs']
        if hparams.get('use_sing_spenv_reduced', False):
            speech_spenvs = sample['speech']['spenvs_reduced']
        sing_mels = sample['sing']['mels']  # [B, T_s, 80]
        sing_spenvs = sample['sing']['spenvs'] if hparams.get('use_sing_spenv', False) else None
        if hparams.get('use_sing_spenv_reduced', False):
            sing_spenvs = sample['sing']['spenvs_reduced']
        speech_mels = sample['speech']['mels']
        speech_mels_mono = sample['speech']['mels_mono']
        source_mels = speech_mels_mono
        if (not hparams.get("use_speech_mel_mono", True)) and hparams.get("use_speech_mel", False):
            source_mels = speech_mels
        if hparams.get("use_speech_wav2vec2_mono", False):
            speech_wav2vec2s_mono = sample['speech']['wav2vec2s_mono']
            source_mels = speech_wav2vec2s_mono
        sing_wav2vec2s = sample['sing']['wav2vec2s'] if hparams.get("use_sing_wav2vec2", False) else None
        if hparams.get("use_sing_input", False):
            source_mels = sing_wav2vec2s if hparams.get("use_sing_wav2vec2", False) else sing_mels
        mel2ph = sample['sing']['mel2ph']
        f0 = sample['sing']['f0']
        uv = sample['sing']['uv']
        energy = sample['sing']['energy']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')

        outputs['losses'] = {}

        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(source_mels, f0, speech_spenvs, spk_embed, ref_mels=None, ref_spenv=sing_spenvs,
                                   txt_tokens=txt_tokens, mel2ph=mel2ph, uv=uv, energy=energy, infer=True, prev_tokens=prev_tokens)
            model_out_wo_gt_spenvs = self.model(source_mels, f0, speech_spenvs, spk_embed, ref_mels=None,
                                                ref_spenv=None, txt_tokens=txt_tokens, mel2ph=mel2ph, uv=uv,
                                                energy=energy, infer=True, prev_tokens=prev_tokens)

            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                gt_f0 = self.pe(sample['sing']['mels'], f0_mean=hparams['sing_f0_mean'],
                                f0_std=hparams['sing_f0_std'])['f0_denorm_pred']  # pe predict from GT mel
                pred_f0 = self.pe(model_out['mel_out'], f0_mean=hparams['sing_f0_mean'],
                                  f0_std=hparams['sing_f0_std'])['f0_denorm_pred']  # pe predict from Pred mel
            else:
                # gt_f0 = denorm_f0(sample['sing']['f0'], sample['sing']['uv'], hparams,
                #                   mean=hparams['sing_f0_mean'], std=hparams['sing_f0_std'])
                # pred_f0 = model_out.get('f0_denorm')
                gt_f0 = pred_f0 = sample['sing']['f0']  # both use gt version
            if hparams.get('plot_wav', False):
                if self.vocoder is None:
                    self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()
                self.plot_wav(batch_idx, sample['sing']['mels'], model_out['mel_out'], is_mel=True, gt_f0=gt_f0, f0=pred_f0)
                self.plot_wav(batch_idx, sample['sing']['mels'], model_out_wo_gt_spenvs['mel_out'], is_mel=True,
                              gt_f0=gt_f0, f0=pred_f0, name="wo_gt_spenvs")
            self.plot_mel(batch_idx, sample['sing']['mels'], model_out['mel_out'], name=f'diffmel_{batch_idx}')
            self.plot_mel(batch_idx, sample['sing']['mels'], model_out_wo_gt_spenvs['mel_out'],
                          name=f'diffmel_wo_gt_spenv_{batch_idx}')
            if hparams.get('diffvae_decoder', 'diffsinger') == 'diffsinger':
                self.plot_mel(batch_idx, sample['sing']['mels'], model_out['fs2_mel'], name=f'fs2mel_{batch_idx}')
            if hparams.get('plot_rhythm', False):
                self.plot_rhythm(batch_idx, model_out, sample, ref_spenvs=sing_spenvs, f0=f0)
            # plot speech and sing mel pairs
            sing_mels = sample['sing']['mels']
            if source_mels.size(1) < sing_mels.size(1):
                source_mels = F.pad(source_mels, (0, 0, 0, sing_mels.size(1)-source_mels.size(1)))
            else:
                sing_mels = F.pad(sing_mels, (0, 0, 0, source_mels.size(1) - sing_mels.size(1)))
            self.plot_mel(batch_idx, sing_mels, source_mels, name=f'speech&sing_mel_{batch_idx}')
            # plot attention map
            if hparams.get('plot_attn_diffvae', False) and 'diffvae_aw_main' in model_out:
                self.plot_matrix(batch_idx, model_out['diffvae_aw_main']**(1/4), name=f'diffvae_aw_main_{batch_idx}')
        return outputs

    def plot_rhythm(self, batch_idx, model_out, sample, ref_spenvs=None, f0=None):
        def r_code_to_figure(r_gt, r_pred, ref_spenvs=None, f0=None):
            fig = plt.figure()
            r_gt = r_gt.cpu().numpy()[0]
            r_pred = r_pred.cpu().numpy()[0]
            plt.plot(r_gt, color='b', label='gt', alpha=0.8)
            plt.plot(r_pred, color='red', label='pred', linewidth=3, alpha=0.5)

            if ref_spenvs is not None and hparams.get('use_sing_spenv_reduced', False):     # plot spenvs reduced
                ref_spenvs = ref_spenvs.cpu().numpy()[0]
                ref_spenvs = ref_spenvs * (hparams.get("rhythm_n_embed", 8) - 1)
                plt.plot(ref_spenvs, color='yellow', label='ref_spenvs', alpha=0.8)

            if f0 is not None:
                f0 = f0.cpu().numpy()[0]
                f0 = (f0 - min(f0)) / (max(f0) - min(f0))
                f0 = f0 * (hparams.get("rhythm_n_embed", 8) - 1)
                plt.plot(f0, color='pink', label='f0')

            plt.legend()
            return fig

        if hparams.get('use_sing_spenv', False) and 'spenv_pred' in model_out:
            self.plot_mel(batch_idx, sample['sing']['spenvs'], model_out['spenv_pred'], name=f'spenv_{batch_idx}')
        if hparams.get('use_sing_spenv', False) and 'q_rhythm' in model_out:  # add quantized rhythm visual
            self.plot_mel(batch_idx, sample['sing']['spenvs'], model_out['q_rhythm'], name=f'q_rhythm_{batch_idx}')

        self.logger.add_figure(f'r_code_gt_pred_{batch_idx}',
                               r_code_to_figure(model_out['r_code_gt_pred'][0],
                                                model_out['r_code_gt_pred'][1], ref_spenvs, f0), self.global_step)

    def test_step(self, sample, batch_idx):
        txt_tokens = sample['txt_tokens']
        speech_spenvs = sample['speech']['spenvs']
        if hparams.get('use_sing_spenv_reduced', False):
            speech_spenvs = sample['speech']['spenvs_reduced']
        sing_spenvs = None
        speech_mels_mono = sample['speech']['mels_mono']
        speech_mels = sample['speech']['mels']
        source_mels = speech_mels_mono
        if (not hparams.get("use_speech_mel_mono", True)) and hparams.get("use_speech_mel", False):
            source_mels = speech_mels
        if hparams.get("use_speech_wav2vec2_mono", False):
            speech_wav2vec2s_mono = sample['speech']['wav2vec2s_mono']
            source_mels = speech_wav2vec2s_mono
        sing_wav2vec2s = sample['sing']['wav2vec2s'] if hparams.get("use_sing_wav2vec2", False) else None
        if hparams.get("use_sing_input", False):
            source_mels = sing_wav2vec2s if hparams.get("use_sing_wav2vec2", False) else sing_mels
        energy = sample['sing']['energy']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        mel2ph, uv, f0 = None, None, None
        ref_mels = None
        if hparams['profile_infer']:
            pass
        else:
            if hparams['use_gt_dur']:
                mel2ph = sample['sing']['mel2ph']
            if hparams['use_gt_f0']:
                f0 = sample['sing']['f0']
                uv = sample['sing']['uv']
                if hparams.get('debug', False):
                    print('Here using gt f0!!')
                    print('test_step f0---------test_step f0---------')
                    print(f0)
            if hparams.get('use_gt_sing_spenv'):
                sing_spenvs = sample['sing']['spenvs'] if hparams.get('use_sing_spenv', False) else None
                if hparams.get('use_sing_spenv_reduced', False):
                    sing_spenvs = sample['sing']['spenvs_reduced']
            if hparams.get('use_midi') is not None and hparams['use_midi']:
                pitch_midi = sample['pitch_midi']
                if hparams['infer'] and hparams.get('pitch_shift', False):
                    pitch_midi = midi_pitch_shift(pitch_midi, hparams.get('avg_f0', 240))
                sample['avg_f0'] = midi_to_hz(pitch_midi)[pitch_midi != 0].mean()
                outputs = self.model(
                    source_mels, f0, speech_spenvs, spk_embed, ref_mels=ref_mels, ref_spenv=sing_spenvs,
                    txt_tokens=txt_tokens, mel2ph=mel2ph, uv=uv, energy=energy, infer=True, pitch_midi=pitch_midi,
                    midi_dur=sample.get('midi_dur'), is_slur=sample.get('is_slur'))
            else:
                outputs = self.model(source_mels, f0, speech_spenvs, spk_embed, ref_mels=ref_mels,
                                     ref_spenv=sing_spenvs, txt_tokens=txt_tokens, mel2ph=mel2ph, uv=uv,
                                     energy=energy, infer=True)
            sample['outputs'] = self.model.out2mel(outputs['mel_out'])
            sample['mel2ph_pred'] = outputs['mel2ph'] if not hparams['use_gt_dur'] and 'mel2ph' in outputs else None
            sample['diffvae_aw_main'] = outputs['diffvae_aw_main'] if hparams.get('plot_attn_diffvae', False) and \
                                                                      'diffvae_aw_main' in outputs else None
            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                sample['f0'] = self.pe(sample['sing']['mels'], f0_mean=hparams['sing_f0_mean'],
                                       f0_std=hparams['sing_f0_std'])['f0_denorm_pred']  # pe predict from GT mel
                sample['f0_pred'] = self.pe(sample['outputs'], f0_mean=hparams['sing_f0_mean'],
                                            f0_std=hparams['sing_f0_std'])['f0_denorm_pred']  # pe predict from Pred mel
            else:
                sample['f0'] = sample['sing']['f0']
                sample['f0_pred'] = sample['sing']['f0']
            return self.after_infer(sample)

    def after_infer(self, predictions, sil_start_frame=0):
        if self.saving_result_pool is None and not hparams['profile_infer']:
            self.saving_result_pool = Pool(min(int(os.getenv('N_PROC', os.cpu_count())), 16))
            self.saving_results_futures = []
        predictions = utils.unpack_dict_to_list(predictions)
        t = tqdm(predictions)
        for num_predictions, prediction in enumerate(t):
            for k, v in prediction.items():
                if type(v) is torch.Tensor:
                    prediction[k] = v.cpu().numpy()
                elif isinstance(v, dict):
                    for k1, v1 in v.items():
                        prediction[k][k1] = v1.cpu().numpy()

            item_name = prediction.get('item_name')
            text = prediction.get('text').replace(":", "%3A")[:80]

            # remove paddings
            mel_gt = prediction['sing']["mels"]
            mel_gt_mask = np.abs(mel_gt).sum(-1) > 0
            mel_gt = mel_gt[mel_gt_mask]
            mel2ph_gt = prediction['sing'].get("mel2ph")
            mel2ph_gt = mel2ph_gt[mel_gt_mask] if mel2ph_gt is not None else None
            mel_pred = prediction["outputs"]
            mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
            mel_pred = mel_pred[mel_pred_mask]
            mel_gt = np.clip(mel_gt, hparams['mel_vmin'], hparams['mel_vmax'])
            mel_pred = np.clip(mel_pred, hparams['mel_vmin'], hparams['mel_vmax'])

            mel2ph_pred = prediction.get("mel2ph_pred")
            if mel2ph_pred is not None:
                if len(mel2ph_pred) > len(mel_pred_mask):
                    mel2ph_pred = mel2ph_pred[:len(mel_pred_mask)]
                mel2ph_pred = mel2ph_pred[mel_pred_mask]

            f0_gt = prediction.get("f0")
            f0_pred = prediction.get("f0_pred")
            if f0_pred is not None:
                f0_gt = f0_gt[mel_gt_mask]
                if len(f0_pred) > len(mel_pred_mask):
                    f0_pred = f0_pred[:len(mel_pred_mask)]
                f0_pred = f0_pred[mel_pred_mask]

            aw_main = prediction.get('diffvae_aw_main', None)

            str_phs = None
            if self.phone_encoder is not None and 'txt_tokens' in prediction:
                str_phs = self.phone_encoder.decode(prediction['txt_tokens'], strip_padding=True)
            gen_dir = os.path.join(hparams['work_dir'],
                                   f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
            if hparams.get('vocoder_use_f0', False):
                try:
                    wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
                except RuntimeError as err:
                    traceback.print_exc()
                    print("f0.shape", f0_pred.shape)
                    print(f"Skip [P] {item_name}")
                    continue
            else:
                wav_pred = self.vocoder.spec2wav(mel_pred)
            if not hparams['profile_infer']:
                os.makedirs(gen_dir, exist_ok=True)
                os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
                os.makedirs(f'{gen_dir}/plot', exist_ok=True)
                os.makedirs(os.path.join(hparams['work_dir'], 'P_mels_npy'), exist_ok=True)
                os.makedirs(os.path.join(hparams['work_dir'], 'G_mels_npy'), exist_ok=True)
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_pred, mel_pred, 'P', item_name, text, gen_dir, str_phs, mel2ph_pred, f0_gt, f0_pred, aw_main]))

                if mel_gt is not None and hparams['save_gt']:
                    if hparams.get('vocoder_use_f0', False):
                        try:
                            wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
                        except RuntimeError as err:
                            traceback.print_exc()
                            print("f0.shape", f0_gt.shape)
                            print(f"Skip [G] {item_name}")
                            continue
                    else:
                        wav_gt = self.vocoder.spec2wav(mel_gt)
                    self.saving_results_futures.append(
                        self.saving_result_pool.apply_async(self.save_result, args=[
                            wav_gt, mel_gt, 'G', item_name, text, gen_dir, str_phs, mel2ph_gt, f0_gt, f0_pred, aw_main]))
                    if hparams['save_f0']:
                        import matplotlib.pyplot as plt
                        # f0_pred_, _ = get_pitch(wav_pred, mel_pred, hparams)
                        f0_pred_ = f0_pred
                        f0_gt_, _ = get_pitch(wav_gt, mel_gt, hparams)
                        fig = plt.figure()
                        plt.plot(f0_pred_, label=r'$f0_P$')
                        plt.plot(f0_gt_, label=r'$f0_G$')
                        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                            pass
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(f'{gen_dir}/plot/[F0][{item_name}]{text}.png', format='png')
                        plt.close(fig)

                t.set_description(
                    f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
            else:
                if 'gen_wav_time' not in self.stats:
                    self.stats['gen_wav_time'] = 0
                self.stats['gen_wav_time'] += len(wav_pred) / hparams['audio_sample_rate']
                print('gen_wav_time: ', self.stats['gen_wav_time'])

        return {}

    def plot_wav(self, batch_idx, gt_wav, wav_out, is_mel=False, gt_f0=None, f0=None, name=''):
        gt_wav = gt_wav[0].cpu().numpy()
        wav_out = wav_out[0].cpu().numpy()
        gt_f0 = gt_f0[0].cpu().numpy()
        f0 = f0[0].cpu().numpy()
        if is_mel:
            gt_wav = self.vocoder.spec2wav(gt_wav, f0=gt_f0)
            wav_out = self.vocoder.spec2wav(wav_out, f0=f0)
        self.logger.add_audio(f'gt_{batch_idx}', gt_wav, sample_rate=hparams['audio_sample_rate'],
                                         global_step=self.global_step)
        name = "_" + name if name != "" else ""
        self.logger.add_audio(f'wav{name}_{batch_idx}', wav_out, sample_rate=hparams['audio_sample_rate'],
                                         global_step=self.global_step)

    @staticmethod
    def save_result(wav_out, mel, prefix, item_name, text, gen_dir, str_phs=None, mel2ph=None, gt_f0=None,
                    pred_f0=None, aw_main=None):
        item_name = item_name.replace('/', '-')
        base_fn = f'[{item_name}][{prefix}]'

        if text is not None:
            base_fn += text
        base_fn += ('-' + hparams['exp_name'])
        np.save(os.path.join(hparams['work_dir'], f'{prefix}_mels_npy', item_name), mel)
        audio.save_wav(wav_out, f'{gen_dir}/wavs/{base_fn}.wav', hparams['audio_sample_rate'],
                       norm=hparams['out_wav_norm'])
        fig = plt.figure(figsize=(14, 10))
        spec_vmin = hparams['mel_vmin']
        spec_vmax = hparams['mel_vmax']
        heatmap = plt.pcolor(mel.T, vmin=spec_vmin, vmax=spec_vmax)
        fig.colorbar(heatmap)
        f0, _ = get_pitch(wav_out, mel, hparams)
        f0 = f0 / 10 * (f0 > 0)
        plt.plot(f0, c='white', linewidth=1, alpha=0.6)
        plt.plot(gt_f0 / 10 * (gt_f0 > 0), c='red', linewidth=1, alpha=0.6)
        if mel2ph is not None and str_phs is not None:
            decoded_txt = str_phs.split(" ")
            dur = mel2ph_to_dur(torch.LongTensor(mel2ph)[None, :], len(decoded_txt))[0].numpy()
            dur = [0] + list(np.cumsum(dur))
            for i in range(len(dur) - 1):
                shift = (i % 20) + 1
                plt.text(dur[i], shift, decoded_txt[i])
                plt.hlines(shift, dur[i], dur[i + 1], colors='b' if decoded_txt[i] != '|' else 'black')
                plt.vlines(dur[i], 0, 5, colors='b' if decoded_txt[i] != '|' else 'black',
                           alpha=1, linewidth=1)
        plt.tight_layout()
        plt.savefig(f'{gen_dir}/plot/{base_fn}.png', format='png', dpi=1000)
        plt.close(fig)
        # plot attn
        if hparams.get('plot_attn_diffvae', False) and aw_main is not None:
            if isinstance(aw_main, torch.Tensor):
                aw_main = aw_main.cpu().numpy()**(1/4)
            fig = plt.figure(figsize=(12, 6))
            c = plt.pcolor(aw_main.T)
            fig.colorbar(c)
            plt.savefig(f'{gen_dir}/plot/{base_fn}_aw.png', format='png', dpi=1000)
            plt.close(fig)

    def add_vqvae_loss(self, output, losses):
        losses['rq_loss'] = output['rq_loss']
        if self.global_step > hparams.get('rhythm_pred_act_steps', -1):    # late start
            losses['r_loss'] = output['r_loss'] * hparams['lambda_rhythm_pred']
        else:
            losses['r_loss'] = 0.0

    def plot_model(self):
        # dummy_input = [torch.rand(1, 300, 80), torch.rand(1, 300), torch.rand(1, 300, 80),
        #                torch.rand(1, 256), torch.rand(1, 300, 80), torch.rand(1, 300, 80)]
        # for i in range(len(dummy_input)):
        #     dummy_input[i] = dummy_input[i].cuda()
        # self.logger.add_graph(self.model, tuple(dummy_input), use_strict_trace=False)
        pass

    def plot_matrix(self, batch_idx, matrix, name=None):
        name = f'matrix_{batch_idx}' if name is None else name
        self.logger.add_figure(name, spec_to_figure(matrix[0]), self.global_step)

    def build_token_encoder(self, data_dir):
        bpe_list_file = os.path.join(data_dir, 'bpe_set.json')
        bpe_list = json.load(open(bpe_list_file))
        return TokenTextEncoder(None, vocab_list=bpe_list, replace_oov='<UNK>')

