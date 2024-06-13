import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing.pool import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt

import utils
from utils.hparams import hparams
from data_gen.tts.data_gen_utils import get_pitch
from utils.indexed_datasets import IndexedDataset
from utils.pitch_utils import norm_interp_f0, denorm_f0, midi_pitch_shift, midi_to_hz, get_uv
import utils.audio as audio
from utils.plot import spec_to_figure
from tasks.base_task import BaseDataset
from tasks.tts.fs2 import FastSpeech2Task

from modules.diffusion.net import DiffNet
from tasks.speech2singing.diffvae import DiffVAETask, MODELS, Speech2SingingDataset
from modules.fastspeech.pe import PitchExtractor
from modules.fastspeech.tts_modules import mel2ph_to_dur
from modules.fastspeech.multi_window_disc import Discriminator
from utils.common_schedulers import RSQRTSchedule

class Sp2SingAdvTask(DiffVAETask):
    def build_model(self):
        self.build_tts_model()
        if hparams['load_ckpt'] != '':
            self.load_ckpt(hparams['load_ckpt'], strict=True)
        utils.print_arch(self.model, 'Generator')
        self.build_disc_model()
        if not hasattr(self, 'gen_params'):
            # self.gen_params = list(self.model.parameters())
            self.gen_params = []
            for params in self.model.parameters():
                if params.requires_grad is True:
                    self.gen_params.append(params)
        return self.model

    def build_disc_model(self):
        self.disc_params = []
        self.use_cond_disc = hparams['use_cond_disc']
        if hparams['mel_gan']:
            disc_win_num = hparams['disc_win_num']
            h = hparams['mel_disc_hidden_size']
            self.mel_disc = Discriminator(
                time_lengths=[32, 64, 128][:disc_win_num],
                # time_lengths=[64, 128, 256][:disc_win_num],
                freq_length=80, hidden_size=h, kernel=(3, 3),
                cond_size=hparams['hidden_size'] if self.use_cond_disc else 0,
                sn=hparams['disc_sn'], reduction=hparams['disc_reduction']
            )
            self.disc_params += list(self.mel_disc.parameters())
            utils.print_arch(self.mel_disc, model_name='Mel Disc')

    def _training_step(self, sample, batch_idx, optimizer_idx):
        log_outputs = {}
        loss_weights = {}
        disc_start = hparams['mel_gan'] and self.global_step > hparams['disc_start_steps'] and \
            hparams['mel_lambda_adv'] > 0

        sing_mels = sample['sing']['mels']  # [B, T_s, 80]
        speech_spenvs = sample['speech']['spenvs']
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
        f0 = sample['sing']['f0']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')

        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            log_outputs, model_out = self.run_model(self.model, sample, return_output=True)
            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            self.disc_cond_gt = self.model_out['decoder_inp'].detach() if hparams['use_cond_disc'] else None
            if disc_start:
                if hparams['infer_adv']:
                    self.model_out = model_out = self.model(source_mels, f0, speech_spenvs, spk_embed)
                self.disc_cond = disc_cond = self.model_out['decoder_inp'].detach() \
                    if hparams['use_cond_disc'] else None
                mel_p = self.model.out2mel(model_out['mel_out'])
                o_ = self.mel_disc(mel_p, disc_cond)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    log_outputs['a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                    loss_weights['a'] = hparams['mel_lambda_adv']
                if pc_ is not None:
                    log_outputs['ac'] = self.mse_loss_fn(pc_, pc_.new_ones(pc_.size()))
                    loss_weights['ac'] = hparams['mel_lambda_adv']
        else:
            #######################
            #    Discriminator    #
            #######################
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                if hparams['rerun_gen']:
                    with torch.no_grad():
                        _, model_out = self.run_model(self.model, sample, return_output=True)
                else:
                    model_out = self.model_out_gt
                mel_g = sample['sing']['mels']
                mel_p = self.model.out2mel(model_out['mel_out'])
                o = self.mel_disc(mel_g, self.disc_cond_gt)
                p, pc = o['y'], o['y_c']
                o_ = self.mel_disc(mel_p, self.disc_cond)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    log_outputs["r"] = self.mse_loss_fn(p, p.new_ones(p.size())) * 2
                    log_outputs["f"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
                if pc_ is not None:
                    log_outputs["rc"] = self.mse_loss_fn(pc, pc.new_ones(pc.size()))
                    log_outputs["fc"] = self.mse_loss_fn(pc_, pc_.new_zeros(pc_.size()))
            if len(log_outputs) == 0:
                return None
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in log_outputs.items()])
        log_outputs['batch_size'] = sample['id'].size()[0]
        log_outputs['lr'] = self.scheduler['gen'].get_lr()
        return total_loss, log_outputs

    def configure_optimizers(self):
        if not hasattr(self, 'gen_params'):
            self.gen_params = list(self.model.parameters())
        optimizer_gen = torch.optim.AdamW(
            self.gen_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        optimizer_disc = torch.optim.AdamW(
            self.disc_params,
            lr=hparams['disc_lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None
        self.scheduler = self.build_scheduler({'gen': optimizer_gen, 'disc': optimizer_disc})
        self.first_valid = True
        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return {
            "gen": RSQRTSchedule(optimizer['gen']),
            "disc": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer["disc"],
                **hparams["discriminator_scheduler_params"]) if optimizer["disc"] is not None else None,
        }

    def on_before_optimization(self):
        if self.opt_idx == 0:
            nn.utils.clip_grad_norm_(self.gen_params, hparams['generator_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.disc_params, hparams["discriminator_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if optimizer_idx == 0:
            self.scheduler['gen'].step(self.global_step)
        else:
            self.scheduler['disc'].step(max(self.global_step - hparams["disc_start_steps"], 1))
