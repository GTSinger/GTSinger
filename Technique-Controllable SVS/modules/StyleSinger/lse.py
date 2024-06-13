from torch import nn
import copy
import torch
from utils.hparams import hparams
from modules.StyleSinger.wavenet import WN
from modules.StyleSinger.RQ import RQBottleneck
import math

from modules.fastspeech.tts_modules import LayerNorm
import torch.nn.functional as F
from utils.tts_utils import group_hidden_by_segs, sequence_mask

from torch.nn import functional as F


class CrossAttenLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CrossAttenLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, src, local_emotion, emotion_key_padding_mask=None, forcing=False):
        # src: (Tph, B, 256) local_emotion: (Temo, B, 256) emotion_key_padding_mask: (B, Temo)
        if forcing:
            maxlength = src.shape[0]
            k = local_emotion.shape[0] / src.shape[0]
            lengths1 = torch.ceil(torch.tensor([i for i in range(maxlength)]).to(src.device) * k) + 1
            lengths2 = torch.floor(torch.tensor([i for i in range(maxlength)]).to(src.device) * k) - 1
            mask1 = sequence_mask(lengths1, local_emotion.shape[0])
            mask2 = sequence_mask(lengths2, local_emotion.shape[0])
            mask = mask1.float() - mask2.float()
            attn_emo = mask.repeat(src.shape[1], 1, 1) # (B, Tph, Temo)
            src2 = torch.matmul(local_emotion.permute(1, 2, 0), attn_emo.float().transpose(1, 2)).permute(2, 0, 1)
        else:
            src2, attn_emo = self.multihead_attn(src, local_emotion, local_emotion, key_padding_mask=emotion_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_emo


class ProsodyAligner(nn.Module):
    def __init__(self, num_layers, guided_sigma=0.3, guided_layers=None, norm=None):
        super(ProsodyAligner, self).__init__()
        self.layers = nn.ModuleList([CrossAttenLayer(d_model=hparams['hidden_size'], nhead=2) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.guided_sigma = guided_sigma
        self.guided_layers = guided_layers if guided_layers is not None else num_layers

    def forward(self, src, local_emotion, src_key_padding_mask=None, emotion_key_padding_mask=None, forcing=False):
        output = src
        guided_loss = 0
        attn_emo_list = []
        for i, mod in enumerate(self.layers):
            # output: (Tph, B, 256), global_emotion: (1, B, 256), local_emotion: (Temo, B, 256) mask: None, src_key_padding_mask: (B, Tph),
            # emotion_key_padding_mask: (B, Temo)
            output, attn_emo = mod(output, local_emotion, emotion_key_padding_mask=emotion_key_padding_mask, forcing=forcing)
            attn_emo_list.append(attn_emo.unsqueeze(1))
            # attn_emo: (B, Tph, Temo) attn: (B, Tph, Tph)
            if i < self.guided_layers and src_key_padding_mask is not None:
                s_length = (~src_key_padding_mask).float().sum(-1) # B
                emo_length = (~emotion_key_padding_mask).float().sum(-1)
                attn_w_emo = _make_guided_attention_mask(src_key_padding_mask.size(-1), s_length, emotion_key_padding_mask.size(-1), emo_length, self.guided_sigma)

                g_loss_emo = attn_emo * attn_w_emo  # N, L, S
                non_padding_mask = (~src_key_padding_mask).unsqueeze(-1) & (~emotion_key_padding_mask).unsqueeze(1)
                guided_loss = g_loss_emo[non_padding_mask].mean() + guided_loss

        if self.norm is not None:
            output = self.norm(output)

        return output, guided_loss, attn_emo_list

def _make_guided_attention_mask(ilen, rilen, olen, rolen, sigma):
    grid_x, grid_y = torch.meshgrid(torch.arange(ilen, device=rilen.device), torch.arange(olen, device=rolen.device))
    grid_x = grid_x.unsqueeze(0).expand(rilen.size(0), -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(rolen.size(0), -1, -1)
    rilen = rilen.unsqueeze(1).unsqueeze(1)
    rolen = rolen.unsqueeze(1).unsqueeze(1)
    return 1.0 - torch.exp(
        -((grid_y.float() / rolen - grid_x.float() / rilen) ** 2) / (2 * (sigma ** 2))
    )

class LocalStyleAdaptor(nn.Module):
    def __init__(self, hidden_size, num_rq_codes=64, padding_idx=0):
        super(LocalStyleAdaptor, self).__init__()
        self.encoder = ConvBlocks(80, hidden_size, [1] * 5, 5, dropout=hparams['vae_dropout'])
        self.n_embed = num_rq_codes
        self.rqvae = RQBottleneck([1,hidden_size],[1,self.n_embed],self.n_embed,hparams['rq_depth'])
        self.wavenet = WN(hidden_channels=80, gin_channels=80, kernel_size=3, dilation_rate=1, n_layers=4)
        self.padding_idx = padding_idx
        self.hidden_size = hidden_size

    def forward(self, ref_mels, ref_f0=None, mel2ph=None, no_rq=False):
        """

        :param ref_mels: [B, T, 80]
        :return: [B, 1, H]
        """
        padding_mask = ref_mels[:, :, 0].eq(self.padding_idx).data
        ref_mels = self.wavenet(ref_mels.transpose(1, 2), x_mask=(~padding_mask).unsqueeze(1).repeat([1, 80, 1])).transpose(1, 2)
        if mel2ph is not None:
            ref_ph, _ = group_hidden_by_segs(ref_mels, mel2ph, torch.max(mel2ph))
            if ref_f0 is not None:
                ref_f0 = ref_f0.unsqueeze(ref_f0.dim()).repeat([1, 1, 80])
                ref_f0,_=group_hidden_by_segs(ref_f0, mel2ph, torch.max(mel2ph))
                ref_ph+=ref_f0

        else:
            ref_ph = ref_mels
            if ref_f0 is not None:
                ref_f0 = ref_f0.unsqueeze(ref_f0.dim()).repeat([1, 1, 80])
                ref_ph+=ref_f0
        
        style = self.encoder(ref_ph)
        if no_rq:
            return style
        z, rq_loss = self.rqvae(style)
        rq_loss = rq_loss.mean()
        return z, rq_loss


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Conv1d(nn.Conv1d):
    """A wrapper around nn.Conv1d, that works on (batch, time, channels)"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, bias=True, padding=0):
        super(Conv1d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride, dilation=dilation,
                                     groups=groups, bias=bias, padding=padding)

    def forward(self, x):
        return super().forward(x.transpose(2, 1)).transpose(2, 1)


def init_weights_func(m):
    classname = m.__class__.__name__
    if classname.find("Conv1d") != -1:
        torch.nn.init.xavier_uniform_(m.weight)


class ResidualBlock(nn.Module):
    """Implements conv->PReLU->norm n-times"""

    def __init__(self, channels, kernel_size, dilation, n=2, norm_type='bn', dropout=0.0,
                 c_multiple=2, ln_eps=1e-12):
        super(ResidualBlock, self).__init__()

        if norm_type == 'bn':
            norm_builder = lambda: nn.BatchNorm1d(channels)
        elif norm_type == 'in':
            norm_builder = lambda: nn.InstanceNorm1d(channels, affine=True)
        elif norm_type == 'gn':
            norm_builder = lambda: nn.GroupNorm(8, channels)
        elif norm_type == 'ln':
            norm_builder = lambda: LayerNorm(channels, dim=1, eps=ln_eps)
        else:
            norm_builder = lambda: nn.Identity()

        self.blocks = [
            nn.Sequential(
                norm_builder(),
                nn.Conv1d(channels, c_multiple * channels, kernel_size, dilation=dilation,
                          padding=(dilation * (kernel_size - 1)) // 2),
                LambdaLayer(lambda x: x * kernel_size ** -0.5),
                nn.GELU(),
                nn.Conv1d(c_multiple * channels, channels, 1, dilation=dilation),
            )
            for i in range(n)
        ]

        self.blocks = nn.ModuleList(self.blocks)
        self.dropout = dropout

    def forward(self, x):
        nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        for b in self.blocks:
            x_ = b(x)
            if self.dropout > 0 and self.training:
                x_ = F.dropout(x_, self.dropout, training=self.training)
            x = x + x_
            x = x * nonpadding
        return x


class ConvBlocks(nn.Module):
    """Decodes the expanded phoneme encoding into spectrograms"""

    def __init__(self, channels, out_dims, dilations, kernel_size,
                 norm_type='ln', layers_in_block=2, c_multiple=2,
                 dropout=0.0, ln_eps=1e-5, init_weights=True):
        super(ConvBlocks, self).__init__()
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels, kernel_size, d,
                            n=layers_in_block, norm_type=norm_type, c_multiple=c_multiple,
                            dropout=dropout, ln_eps=ln_eps)
              for d in dilations],
        )
        if norm_type == 'bn':
            norm = nn.BatchNorm1d(channels)
        elif norm_type == 'in':
            norm = nn.InstanceNorm1d(channels, affine=True)
        elif norm_type == 'gn':
            norm = nn.GroupNorm(8, channels)
        elif norm_type == 'ln':
            norm = LayerNorm(channels, dim=1, eps=ln_eps)
        self.last_norm = norm
        self.post_net1 = nn.Conv1d(channels, out_dims, kernel_size=3, padding=1)
        if init_weights:
            self.apply(init_weights_func)

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return:  [B, T, H]
        """
        x = x.transpose(1, 2)
        nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        x = self.res_blocks(x) * nonpadding
        x = self.last_norm(x) * nonpadding
        x = self.post_net1(x) * nonpadding
        return x.transpose(1, 2)
