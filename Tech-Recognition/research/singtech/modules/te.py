from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.commons.hparams import hparams
from utils.commons.gpu_mem_track import MemTracker
from modules.commons.layers import Embedding
from modules.commons.conv import ResidualBlock, ConvBlocks
from modules.commons.conformer.conformer import ConformerLayers, FastConformerLayers
from research.rme.modules.unet import Unet

# gpu_tracker = MemTracker()

class TechExtractor(nn.Module):
    def __init__(self, hparams):
        super(TechExtractor, self).__init__()
        self.hparams = deepcopy(hparams)
        self.hidden_size = hidden_size = hparams['hidden_size']
        self.dropout = hparams.get('dropout', 0.0)
        self.tech_threshold = hparams.get('tech_threshold', 0.5)

        self.mel_proj = nn.Conv1d(hparams['use_mel_bins'], hidden_size, kernel_size=3, padding=1)
        self.mel_encoder = ConvBlocks(hidden_size, out_dims=hidden_size, dilations=None, kernel_size=3,
                                      layers_in_block=2, c_multiple=1, dropout=self.dropout, num_layers=1,
                                      post_net_kernel=3, act_type='swish')
        self.use_pitch = hparams.get('use_f0', True)
        if self.use_pitch:
            self.pitch_embed = Embedding(300, hidden_size, 0, 'kaiming')
            self.uv_embed = Embedding(3, hidden_size, 0, 'kaiming')
        self.use_breathiness = hparams.get('use_breathiness', False)
        if self.use_breathiness:
            self.breathiness_embed = Embedding(300, hidden_size, 0, 'kaiming')
        self.use_energy = hparams.get('use_energy', False)
        if self.use_energy:
            self.energy_embed = Embedding(300, hidden_size, 0, 'kaiming')
        self.use_zcr = hparams.get('use_zcr', False)
        if self.use_zcr:
            self.zcr_embed = Embedding(300, hidden_size, 0, 'kaiming')
        self.ph_bd_embed = Embedding(3, hidden_size, 0, 'kaiming')
        self.cond_encoder = ConvBlocks(hidden_size, out_dims=hidden_size, dilations=None, kernel_size=3,
                                       layers_in_block=1, c_multiple=1, dropout=self.dropout, num_layers=1,
                                       post_net_kernel=3, act_type='swish')

        # backbone
        updown_rates = [2, 2, 2]
        channel_multiples = [1.125, 1.125, 1.125]
        if hparams.get('updown_rates', None) is not None:
            updown_rates = [int(i) for i in hparams.get('updown_rates', None).split('-')]
        if hparams.get('channel_multiples', None) is not None:
            channel_multiples = [float(i) for i in hparams.get('channel_multiples', None).split('-')]
        assert len(updown_rates) == len(channel_multiples)
        # convs
        if hparams.get('bkb_net', 'conv') == 'conv':
            self.net = Unet(hidden_size, down_layers=len(updown_rates), mid_layers=hparams.get('bkb_layers', 12),
                            up_layers=len(updown_rates), kernel_size=3, updown_rates=updown_rates,
                            channel_multiples=channel_multiples, dropout=0, is_BTC=True,
                            constant_channels=False, mid_net=None, use_skip_layer=hparams.get('unet_skip_layer', False))
        # conformer
        elif hparams.get('bkb_net', 'conv') == 'conformer':
            mid_net = ConformerLayers(
                hidden_size, num_layers=hparams.get('bkb_layers', 2), kernel_size=hparams.get('conformer_kernel', 9),
                dropout=self.dropout, num_heads=4)
            self.net = Unet(hidden_size, down_layers=len(updown_rates), up_layers=len(updown_rates), kernel_size=3,
                            updown_rates=updown_rates, channel_multiples=channel_multiples, dropout=0,
                            is_BTC=True, constant_channels=False, mid_net=mid_net,
                            use_skip_layer=hparams.get('unet_skip_layer', False),
                            skip_scale=hparams.get('unet_skip_scale', 1.0))

        # tech prediction
        self.tech_attn_num_head = hparams.get('tech_attn_num_head', 1)
        self.multihead_dot_attn = nn.Linear(hidden_size, self.tech_attn_num_head)
        self.post = ConvBlocks(hidden_size, out_dims=hidden_size, dilations=None, kernel_size=3,
                               layers_in_block=1, c_multiple=1, dropout=self.dropout, num_layers=1,
                               post_net_kernel=3, act_type='swish')
        self.tech_out = nn.Linear(hidden_size, hparams.get('tech_num', 6))
        self.tech_num = hparams.get('tech_num', 6)
        self.tech_temperature = max(1e-7, hparams.get('tech_temperature', 1.0))

        self.reset_parameters()

    def forward(self, mel=None, ph_bd=None, pitch=None, uv=None, variance=None, non_padding=None, train=True):
        # gpu_tracker.track()
        ret = {}
        bsz, T, _ = mel.shape

        mel_embed = self.mel_proj(mel.transpose(1, 2)).transpose(1, 2)
        mel_embed = self.mel_encoder(mel_embed)
        if self.use_pitch and pitch is not None and uv is not None:
            pitch_embed = self.pitch_embed(pitch) + self.uv_embed(uv)  # [B, T, C]
        else:
            pitch_embed = 0

        variance_embed = 0
        if self.use_breathiness and 'breathiness' in variance:
            variance_embed = variance_embed + self.breathiness_embed(variance['breathiness'])
        if self.use_energy and 'energy' in variance:
            variance_embed = variance_embed + self.energy_embed(variance['energy'])
        if self.use_zcr and 'zcr' in variance:
            variance_embed = variance_embed + self.zcr_embed(variance['zcr'])
        variance_embed = variance_embed / np.sqrt(2)

        feat = self.cond_encoder(mel_embed + pitch_embed + variance_embed)

        feat = self.net(feat)   # [B, T, C]

        # note pitch prediction
        attn = torch.sigmoid(self.multihead_dot_attn(feat))     # [B, T, C] -> [B, T, num_head]
        attn = F.dropout(attn, self.dropout, train)
        attn_feat = feat.unsqueeze(3) * attn.unsqueeze(2)   # [B, T, C, 1] x [B, T, 1, num_head] -> [B, T, C, num_head]
        attn_feat = torch.mean(attn_feat, dim=-1)   # [B, T, C, num_head] -> [B, T, C]
        mel2ph = torch.cumsum(ph_bd, 1)
        ph_length = torch.max(torch.sum(ph_bd, dim=1)).item() + 1  # [B]
        # print('note_length', note_length)

        attn = torch.mean(attn, dim=-1, keepdim=True)   # [B, T, num_head] -> [B, T, 1]
        denom = mel2ph.new_zeros(bsz, ph_length, dtype=attn.dtype).scatter_add_(
            dim=1, index=mel2ph, src=attn.squeeze(-1)
        )  # [B, T] -> [B, ph_length] count the note frames of each note (with padding excluded)
        frame2ph = mel2ph.unsqueeze(-1).repeat(1, 1, self.hidden_size)  # [B, T] -> [B, T, C], with padding included
        ph_aggregate = frame2ph.new_zeros(bsz, ph_length, self.hidden_size, dtype=attn_feat.dtype).scatter_add_(
            dim=1, index=frame2ph, src=attn_feat
        )  # [B, T, C] -> [B, ph_length, C]
        ph_aggregate = ph_aggregate / (denom.unsqueeze(-1) + 1e-5)
        ph_aggregate = F.dropout(ph_aggregate, self.dropout, train)
        tech_logits = self.post(ph_aggregate)
        tech_logits = self.tech_out(tech_logits) / self.tech_temperature
        # tech_logits = torch.clamp(tech_logits, min=-16., max=16.)     # don't know need it or not

        ret['tech_logits'] = tech_logits    # [B, ph_length, note_num]
        tech_pred = torch.sigmoid(tech_logits)  # [B, ph_length, note_num]
        tech_pred = (tech_pred > self.tech_threshold).long()     # [B, ph_length]
        ret['tech_pred'] = tech_pred

        # gpu_tracker.track()
        return ret

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.multihead_dot_attn.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.tech_out.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.mel_proj.weight, mode='fan_in')
        nn.init.constant_(self.multihead_dot_attn.bias, 0.0)
        nn.init.constant_(self.tech_out.bias, 0.0)
