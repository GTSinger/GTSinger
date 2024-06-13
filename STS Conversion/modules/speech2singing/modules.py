import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import EncSALayer, LayerNorm, BatchNorm1dTBC, MultiheadAttention, \
    SinusoidalPositionalEmbedding
from modules.fastspeech.tts_modules import FFTBlocks, DEFAULT_MAX_TARGET_POSITIONS
from utils.tts_utils import sequence_mask


class CrossformerEncoderLayer(EncSALayer):
    def __init__(self, c, num_heads, dropout, kdim=None, vdim=None, attention_dropout=0.1,
                 relu_dropout=0.1, kernel_size=9, padding='SAME', norm='ln', act='gelu', attn_class='base',
                 win_size=0.0, cb_config=None):
        super(CrossformerEncoderLayer, self).__init__(c, num_heads, dropout, attention_dropout,
                                                      relu_dropout, kernel_size, padding, norm, act)
        if norm == 'ln':
            self.layer_norm1 = LayerNorm(c)
        elif norm == 'bn':
            self.layer_norm1 = BatchNorm1dTBC(c)
        self.attn_class = attn_class
        if attn_class == 'base':
            self.cross_attn = MultiheadAttention(
                self.c, num_heads, kdim, vdim, self_attention=False, dropout=attention_dropout, bias=False,
            )
        elif attn_class == 'content_based':
            cb_config = {} if cb_config is None else cb_config
            self.cross_attn = ContentBasedAttention(self.c, num_heads, attention_dropout, **cb_config)
        self.win_size = win_size
        self.self_attn = None

    def forward(self, q, k, v, encoder_padding_mask=None, key_padding_mask=None,
                need_weights=False, forcing=False, **kwargs):
        # input shape: [T, B, C]
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        residual = q
        q = self.layer_norm1(q)
        if forcing:
            maxlength = q.shape[0]
            scale = k.shape[0] / q.shape[0]
            lengths1 = torch.ceil(torch.tensor([i for i in range(maxlength)]).to(q.device) * scale) + 1
            lengths2 = torch.floor(torch.tensor([i for i in range(maxlength)]).to(q.device) * scale) - 1
            mask1 = sequence_mask(lengths1, k.shape[0])
            mask2 = sequence_mask(lengths2, k.shape[0])
            mask = mask1.float() - mask2.float()
            attn_weights = mask.repeat(q.shape[1], 1, 1)  # (B, T1, T2)
            # aw [B, T1, T2] -> [B, T2, T1]
            # k [T2, B, H] -> [B, H, T2]
            # k * aw = x [B, H, T1] -> [T1, B, H]
            x = torch.matmul(k.permute(1, 2, 0), attn_weights.float().transpose(1, 2)).permute(2, 0, 1)
        else:
            if self.attn_class == 'content_based':
                x, attn_weights = self.cross_attn(q, k, v, key_padding_mask, encoder_padding_mask, need_weights)
            else:   # 'base'
                if self.win_size > 0:   # apply attention mask
                    bsz, src_len, tgt_len = q.shape[1], k.shape[0], q.shape[0]
                    key_nonpadding_mask_win = torch.zeros(src_len).to(q.device)  # [BN, Tkv]
                    key_padding_mask_win_TS = torch.zeros((tgt_len, src_len)).to(q.device)
                    for t in range(q.shape[0]):
                        win_size = int(src_len * self.win_size) // 2
                        win_center = int(t / tgt_len * src_len)
                        key_nonpadding_mask_win = key_nonpadding_mask_win.zero_()
                        key_nonpadding_mask_win[max(0, win_center - win_size): win_center + win_size] = 1
                        key_padding_mask_win = (1 - key_nonpadding_mask_win.float()).bool()
                        key_padding_mask_win_TS[t, :] = key_padding_mask_win
                    x, attn_weights, = self.cross_attn(q, k, v, key_padding_mask, need_weights=need_weights,
                                                       attn_mask=key_padding_mask_win_TS.bool())
                else:
                    x, attn_weights, = self.cross_attn(q, k, v, key_padding_mask, need_weights=need_weights)
        x = residual + F.dropout(x, self.dropout, training=self.training)      # residual
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]

        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        if not need_weights:
            attn_weights = None
        return x, attn_weights


class CrossFormerBlocks(nn.Module):
    def __init__(self, hidden_size, num_layers, kv_size=None, ffn_kernel_size=9, dropout=0.0, num_heads=2,
                 use_pos_embed=True, use_last_norm=True, norm='ln', use_pos_embed_alpha=True,
                 num_guided_layers=0, guided_sigma=0.3, attn_class='base', win_size=0.0, cb_config=None):
        super(CrossFormerBlocks, self).__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.kv_size = kv_size
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm
        self.num_guided_layers = num_guided_layers
        self.guided_sigma = guided_sigma
        if use_pos_embed:
            self.max_source_positions = DEFAULT_MAX_TARGET_POSITIONS
            self.padding_idx = 0
            self.pos_embed_alpha = nn.Parameter(torch.Tensor([1])) if use_pos_embed_alpha else 1
            self.embed_positions_1 = SinusoidalPositionalEmbedding(
                embed_dim, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS)
            self.embed_positions_2 = SinusoidalPositionalEmbedding(
                self.kv_size, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            CrossformerEncoderLayer(self.hidden_size, num_heads, self.dropout, kv_size, kv_size,
                                    kernel_size=ffn_kernel_size, attn_class=attn_class,
                                    win_size=win_size, cb_config=cb_config)
            for _ in range(self.num_layers)])

        if self.use_last_norm:
            if norm == 'ln':
                self.layer_norm = nn.LayerNorm(embed_dim)
            elif norm == 'bn':
                self.layer_norm = BatchNorm1dTBC(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, x1, x2, padding_mask=None, attn_mask=None, need_weights=False, forcing=False):
        """
        :param x1: q, [B, T1, C1]   C1: hidden_size
        :param x2: kv, [B, T2, C2]   C2: kv_size
        :param padding_mask: [B, T]
        :return: [B, T1, C] or [B, T1, C1] and [B, T2, C2]
        """
        x1_padding_mask = x1.abs().sum(-1).eq(0).data if padding_mask is None else padding_mask
        x2_padding_mask = x2.abs().sum(-1).eq(0).data if padding_mask is None else padding_mask
        x1_attn_mask = attn_mask
        x1_nonpadding_mask_TB = 1 - x1_padding_mask.transpose(0, 1).float()[:, :, None]  # [T, B, 1]
        x2_nonpadding_mask_TB = 1 - x2_padding_mask.transpose(0, 1).float()[:, :, None]  # [T, B, 1]
        if self.use_pos_embed:
            x1_positions = self.pos_embed_alpha * self.embed_positions_1(x1[..., 0])
            x2_positions = self.pos_embed_alpha * self.embed_positions_2(x2[..., 0])
            x1 = x1 + x1_positions
            x2 = x2 + x2_positions
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
        # B x T x C1 -> T x B x C1
        x1 = x1.transpose(0, 1) * x1_nonpadding_mask_TB
        x2 = x2.transpose(0, 1) * x2_nonpadding_mask_TB
        guided_loss = 0
        for i, layer in enumerate(self.layers):
            x1, aw = layer(x1, x2, x2, encoder_padding_mask=x1_padding_mask, key_padding_mask=x2_padding_mask,
                           attn_mask=x1_attn_mask, need_weights=need_weights, forcing=forcing)
            x1 = x1 * x1_nonpadding_mask_TB
            if i < self.num_guided_layers and need_weights:
                length_1 = (~x1_padding_mask).float().sum(-1)   # [B]
                length_2 = (~x2_padding_mask).float().sum(-1)   # [B]
                aw_mask = _make_guided_attention_mask(x1_padding_mask.size(-1), length_1, x2_padding_mask.size(-1),
                                                      length_2, self.guided_sigma)

                g_loss = aw * aw_mask
                non_padding_mask = (~x1_padding_mask).unsqueeze(-1) & (~x2_padding_mask).unsqueeze(1)
                guided_loss = guided_loss + g_loss[non_padding_mask].mean()
        if self.use_last_norm:
            x1 = self.layer_norm(x1) * x1_nonpadding_mask_TB
        if not need_weights:
            aw = None
        return x1.transpose(0, 1), aw, guided_loss  # x1 [B, T, C], aw1 [B, T1, T2], aw2 [B, T2, T1]


def _make_guided_attention_mask(ilen, rilen, olen, rolen, sigma):
    grid_x, grid_y = torch.meshgrid(torch.arange(ilen, device=rilen.device), torch.arange(olen, device=rolen.device))
    grid_x = grid_x.unsqueeze(0).expand(rilen.size(0), -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(rolen.size(0), -1, -1)
    rilen = rilen.unsqueeze(1).unsqueeze(1)
    rolen = rolen.unsqueeze(1).unsqueeze(1)
    return 1.0 - torch.exp(
        -((grid_y.float() / rolen - grid_x.float() / rilen) ** 2) / (2 * (sigma ** 2))
    )

