import matplotlib
matplotlib.use('Agg')

from tasks.tts.tts_base import TTSBaseTask
# from vocoders.base_vocoder import get_vocoder_cls
from tasks.tts.vocoder_infer.base_vocoder import BaseVocoder, get_vocoder_cls
from tasks.tts.dataset_utils import FastSpeechDataset
from modules.commons.ssim import ssim
import os
from modules.fastspeech.tts_modules import mel2ph_to_dur
from utils.hparams import hparams
from utils.plot import spec_to_figure, dur_to_figure, f0_to_figure
from utils.pitch_utils import denorm_f0
from modules.fastspeech.fs2 import FastSpeech2
import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import utils
import torch.distributions
import numpy as np


class FastSpeech2Task(TTSBaseTask):
    def __init__(self):
        super(FastSpeech2Task, self).__init__()
        self.dataset_cls = FastSpeechDataset
        self.mse_loss_fn = torch.nn.MSELoss()
        mel_losses = hparams['mel_loss'].split("|")
        self.loss_and_lambda = {}
        for i, l in enumerate(mel_losses):
            if l == '':
                continue
            if ':' in l:
                l, lbd = l.split(":")
                lbd = float(lbd)
            else:
                lbd = 1.0
            self.loss_and_lambda[l] = lbd
        print("| Mel losses:", self.loss_and_lambda)
        self.sil_ph = self.phone_encoder.sil_phonemes()
        f0_stats_fn = f'{hparams["binary_data_dir"]}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            hparams['f0_mean'], hparams['f0_std'] = np.load(f0_stats_fn)
            hparams['f0_mean'] = float(hparams['f0_mean'])
            hparams['f0_std'] = float(hparams['f0_std'])

    def build_tts_model(self):
        self.model = FastSpeech2(self.phone_encoder)

    def build_model(self):
        self.build_tts_model()
        if hparams['load_ckpt'] != '':
            self.load_ckpt(hparams['load_ckpt'], strict=False)
        utils.print_arch(self.model)
        return self.model

    def _training_step(self, sample, batch_idx, _):
        loss_output = self.run_model(self.model, sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        return total_loss, loss_output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        mel_out = self.model.out2mel(model_out['mel_out'])
        outputs = utils.tensors_to_scalars(outputs)
        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots']:
            vmin = hparams['mel_vmin']
            vmax = hparams['mel_vmax']
            self.plot_mel(batch_idx, sample['mels'], mel_out)
            self.plot_dur(batch_idx, sample, model_out)
            if hparams['use_pitch_embed']:
                self.plot_pitch(batch_idx, sample, model_out)
            if self.vocoder is None:
                self.vocoder = get_vocoder_cls(hparams)()
            if self.global_step > 0:
                spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
                # with gt duration
                model_out = self.model(sample['txt_tokens'], mel2ph=sample['mel2ph'],
                                       spk_embed=spk_embed, infer=True)
                wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu())
                self.logger.add_audio(f'wav_gtdur_{batch_idx}', wav_pred, self.global_step,
                                      hparams['audio_sample_rate'])
                self.logger.add_figure(
                    f'mel_gtdur_{batch_idx}',
                    spec_to_figure(model_out['mel_out'][0], vmin, vmax), self.global_step)
                # with pred duration
                model_out = self.model(sample['txt_tokens'], spk_embed=spk_embed, infer=True)
                self.logger.add_figure(
                    f'mel_{batch_idx}',
                    spec_to_figure(model_out['mel_out'][0], vmin, vmax), self.global_step)
                wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu())
                self.logger.add_audio(f'wav_{batch_idx}', wav_pred, self.global_step, hparams['audio_sample_rate'])
            # gt wav
            if self.global_step <= hparams['valid_infer_interval']:
                mel_gt = sample['mels'][0].cpu()
                wav_gt = self.vocoder.spec2wav(mel_gt)
                self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, 22050)
        return outputs

    def run_model(self, model, sample, return_output=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        energy = sample['energy']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        output = model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed,
                       ref_mels=target, f0=f0, uv=uv, energy=energy,
                       tgt_mels=target, infer=False)
        losses = {}
        self.add_mel_loss(output['mel_out'], target, losses)
        self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        if hparams['use_pitch_embed']:
            self.add_pitch_loss(output, sample, losses)
        if not return_output:
            return losses
        else:
            return losses, output

    ############
    # losses
    ############
    def add_mel_loss(self, mel_out, target, losses, postfix='', mel_mix_loss=None):
        nonpadding = target.abs().sum(-1).ne(0).float()
        for loss_name, lbd in self.loss_and_lambda.items():
            if 'l1' == loss_name:
                l = self.l1_loss(mel_out, target)
            elif 'mse' == loss_name:
                l = self.mse_loss(mel_out, target)
            elif 'ssim' == loss_name:
                l = self.ssim_loss(mel_out, target)
            elif 'gdl' == loss_name:
                l = self.gdl_loss_fn(mel_out, target, nonpadding) \
                    * self.loss_and_lambda['gdl']
            losses[f'{loss_name}{postfix}'] = l * lbd

    def l1_loss(self, decoder_output, target):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        l1_loss = F.l1_loss(decoder_output, target, reduction='none')
        weights = self.weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def add_energy_loss(self, energy_pred, energy, losses):
        nonpadding = (energy != 0).float()
        loss = (F.mse_loss(energy_pred, energy, reduction='none') * nonpadding).sum() / nonpadding.sum()
        loss = loss * hparams['lambda_energy']
        losses['e'] = loss

    def mse_loss(self, decoder_output, target):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        assert decoder_output.shape == target.shape
        mse_loss = F.mse_loss(decoder_output, target, reduction='none')
        weights = self.weights_nonzero_speech(target)
        mse_loss = (mse_loss * weights).sum() / weights.sum()
        return mse_loss

    def ssim_loss(self, decoder_output, target, bias=6.0):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        assert decoder_output.shape == target.shape
        weights = self.weights_nonzero_speech(target)
        decoder_output = decoder_output[:, None] + bias
        target = target[:, None] + bias
        ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
        ssim_loss = (ssim_loss * weights).sum() / weights.sum()
        return ssim_loss

    def add_dur_loss(self, dur_pred, mel2ph, txt_tokens, losses=None):
        """

        :param dur_pred: [B, T], float, log scale
        :param mel2ph: [B, T]
        :param txt_tokens: [B, T]
        :param losses:
        :return:
        """
        B, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        dur_gt = mel2ph_to_dur(mel2ph, T).float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p in self.sil_ph:
            is_sil = is_sil | (txt_tokens == self.phone_encoder.encode(p)[0])
        is_sil = is_sil.float()  # [B, T_txt]
        losses['pdur'] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction='none')
        losses['pdur'] = (losses['pdur'] * nonpadding).sum() / nonpadding.sum()
        losses['pdur'] = losses['pdur'] * hparams['lambda_ph_dur']
        dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        # use linear scale for sent and word duration
        if hparams['lambda_word_dur'] > 0:
            word_id = (is_sil.cumsum(-1) * (1 - is_sil)).long()
            word_dur_p = dur_pred.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_pred)[:, 1:]
            word_dur_g = dur_gt.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_gt)[:, 1:]
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction='none')
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses['wdur'] = wdur_loss * hparams['lambda_word_dur']
        if hparams['lambda_sent_dur'] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction='mean')
            losses['sdur'] = sdur_loss.mean() * hparams['lambda_sent_dur']

    def add_pitch_loss(self, output, sample, losses):
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (mel2ph != 0).float() if hparams['pitch_type'] == 'frame' \
            else (sample['txt_tokens'] != 0).float()
        self.add_f0_loss(output['pitch_pred'], f0, uv, losses, nonpadding=nonpadding) # output['pitch_pred']: [B, T, 2], f0: [B, T], uv: [B, T]

    def add_f0_loss(self, p_pred, f0, uv, losses, nonpadding, postfix=''):
        assert p_pred[..., 0].shape == f0.shape
        if hparams['use_uv'] and hparams['pitch_type'] == 'frame':
            assert p_pred[..., 1].shape == uv.shape, (p_pred.shape, uv.shape)
            losses[f'uv{postfix}'] = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1], uv, reduction='none') * nonpadding).sum() \
                                     / nonpadding.sum() * hparams['lambda_uv']
            nonpadding = nonpadding * (uv == 0).float()
        f0_pred = p_pred[:, :, 0]
        pitch_loss_fn = F.l1_loss if hparams['pitch_loss'] == 'l1' else F.mse_loss
        losses[f'f0{postfix}'] = (pitch_loss_fn(f0_pred, f0, reduction='none') * nonpadding).sum() \
                                 / nonpadding.sum() * hparams['lambda_f0']


    ############
    # validation plots
    ############
    def plot_dur(self, batch_idx, sample, model_out):
        T_txt = sample['txt_tokens'].shape[1]
        dur_gt = mel2ph_to_dur(sample['mel2ph'], T_txt)[0]
        dur_pred = model_out['dur']
        if hasattr(self.model, 'out2dur'):
            dur_pred = self.model.out2dur(model_out['dur']).float()
        txt = self.phone_encoder.decode(sample['txt_tokens'][0].cpu().numpy())
        txt = txt.split(" ")
        self.logger.add_figure(
            f'dur_{batch_idx}', dur_to_figure(dur_gt, dur_pred, txt), self.global_step)

    def plot_pitch(self, batch_idx, sample, model_out):
        self.logger.add_figure(
            f'f0_{batch_idx}',
            f0_to_figure(model_out['f0_denorm'][0], None, model_out['f0_denorm_pred'][0]),
            self.global_step)

    ############
    # inference
    ############
    def test_step(self, sample, batch_idx):
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        txt_tokens = sample['txt_tokens']
        mel2ph, uv, f0 = None, None, None
        ref_mels = sample['mels']
        if hparams['use_gt_dur']:
            mel2ph = sample['mel2ph']
        if hparams['use_gt_f0']:
            f0 = sample['f0']
            uv = sample['uv']
        run_model = lambda: self.model(
            txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, ref_mels=ref_mels, infer=True)
        if hparams['profile_infer']:
            mel2ph, uv, f0 = sample['mel2ph'], sample['uv'], sample['f0']
            with utils.Timer('fs', enable=True):
                outputs = run_model()
            if 'gen_wav_time' not in self.stats:
                self.stats['gen_wav_time'] = 0
            wav_time = float(outputs["mels_out"].shape[1]) * hparams['hop_size'] / hparams["audio_sample_rate"]
            self.stats['gen_wav_time'] += wav_time
            print(f'[Timer] wav total seconds: {self.stats["gen_wav_time"]}')
            from pytorch_memlab import LineProfiler
            with LineProfiler(self.model.forward) as prof:
                run_model()
            prof.print_stats()
        else:
            outputs = run_model()
            sample['outputs'] = self.model.out2mel(outputs['mel_out'])
            sample['mel2ph_pred'] = outputs['mel2ph']
            if hparams['use_pitch_embed']:
                sample['f0'] = denorm_f0(sample['f0'], sample['uv'], hparams)
                if hparams['pitch_type'] == 'ph':
                    sample['f0'] = torch.gather(F.pad(sample['f0'], [1, 0]), 1, sample['mel2ph'])
                sample['f0_pred'] = outputs.get('f0_denorm')
            return self.after_infer(sample)
