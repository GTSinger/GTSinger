import logging
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.commons.common_layers import ConvNorm, Embedding
from modules.commons.common_layers import SinusoidalPositionalEmbedding, Linear, EncSALayer, DecSALayer, BatchNorm1dTBC
from modules.commons.mixture import sample_from_mixture
from utils.hparams import hparams

DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 2000


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout, kernel_size=None, num_heads=2, norm='ln'):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = EncSALayer(
            hidden_size, num_heads, dropout=dropout,
            attention_dropout=0.0, relu_dropout=dropout,
            kernel_size=kernel_size
            if kernel_size is not None else hparams['enc_ffn_kernel_size'],
            padding=hparams['ffn_padding'],
            norm=norm, act=hparams['ffn_act'])

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout, kernel_size=None, num_heads=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = DecSALayer(
            hidden_size, num_heads, dropout=dropout,
            attention_dropout=0.0, relu_dropout=dropout,
            kernel_size=hparams['dec_ffn_kernel_size'] if kernel_size is None else kernel_size,
            act=hparams['ffn_act'])

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)

    def clear_buffer(self, *args):
        return self.op.clear_buffer(*args)

    def set_buffer(self, *args):
        return self.op.set_buffer(*args)


######################
# fastspeech modules
######################
class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class DurationPredictor(torch.nn.Module):
    """Duration predictor module.
    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`, those are calculated in linear domain.
    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0, padding='SAME'):
        """Initilize duration predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == 'SAME'
                                       else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        if hparams['dur_loss'] in ['mse', 'huber']:
            odims = 1
        elif hparams['dur_loss'] == 'mog':
            odims = 15
        elif hparams['dur_loss'] == 'crf':
            odims = 32
            from torchcrf import CRF
            self.crf = CRF(odims, batch_first=True)
        self.linear = torch.nn.Linear(n_chans, odims)

    def _forward(self, xs, x_masks=None, is_inference=False):
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
            if x_masks is not None:
                xs = xs * (1 - x_masks.float())[:, None, :]

        xs = self.linear(xs.transpose(1, -1))  # [B, T, C]
        xs = xs * (1 - x_masks.float())[:, :, None]  # (B, T, C)
        if is_inference:
            return self.out2dur(xs), xs
        else:
            if hparams['dur_loss'] in ['mse']:
                xs = xs.squeeze(-1)  # (B, Tmax)
        return xs

    def out2dur(self, xs):
        if hparams['dur_loss'] in ['mse']:
            # NOTE: calculate in log domain
            xs = xs.squeeze(-1)  # (B, Tmax)
            dur = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()  # avoid negative value
        elif hparams['dur_loss'] == 'mog':
            dur = sample_from_mixture(xs.transpose(1, 2), greedy=hparams['greedy_sample'],
                                      dist_cls=torch.distributions.Normal)
            dur = torch.round((dur + 1) * 16).clamp(min=0, max=32).long()
        elif hparams['dur_loss'] == 'crf':
            dur = torch.LongTensor(self.crf.decode(xs)).cuda()
        return dur

    def forward(self, xs, x_masks=None):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).
        """
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        """Inference duration.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).
        """
        return self._forward(xs, x_masks, True)


def pad_list(xs, pad_value, max_len=None):
    """Perform padding for the list of tensors.
    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.
    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).
    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    """
    n_batch = len(xs)
    if max_len is None:
        max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, :min(xs[i].size(0), max_len)] = xs[i][:max_len]

    return pad


class LengthRegulator(torch.nn.Module):
    """Length regulator module for feed-forward Transformer.
    This is a module of length regulator described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, pad_value=0.0):
        """Initilize length regulator module.
        Args:
            pad_value (float, optional): Value used for padding.
        """
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self, ds, ilens, alpha=1.0, max_len=None):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (LongTensor): Batch of durations of each frame (B, T).
            ilens (LongTensor): Batch of input lengths (B,).
            alpha (float, optional): Alpha value to control speed of speech.
        Returns:
            Tensor: replicated input tensor based on durations (B, T*, D).
        """
        assert alpha > 0
        if alpha != 1.0:
            ds = torch.round(ds.float() * alpha).long()
        ds = [d[:ilen] for d, ilen in zip(ds, ilens)]
        mel2ph = [self._repeat_one_sequence(torch.arange(len(d)).to(d.device), d) + 1 for d in ds]
        return pad_list(mel2ph, 0, max_len).long()

    def _repeat_one_sequence(self, x, d):
        """Repeat each frame according to duration.
        Examples:
            >>> x = torch.tensor([[1], [2], [3]])
            tensor([[1],
                    [2],
                    [3]])
            >>> d = torch.tensor([1, 2, 3])
            tensor([1, 2, 3])
            >>> self._repeat_one_sequence(x, d)
            tensor([[1],
                    [2],
                    [2],
                    [3],
                    [3],
                    [3]])
        """
        if d.sum() == 0:
            logging.warn("all of the predicted durations are 0. fill 0 with 1.")
            d = d.fill_(1)
        return torch.cat([x_.repeat(int(d_), 1) for x_, d_ in zip(x, d) if d_ != 0], dim=0)


class PitchPredictor(torch.nn.Module):
    def __init__(self, idim, n_layers=5, n_chans=384, odim=2, kernel_size=5,
                 dropout_rate=0.1, padding='SAME'):
        """Initilize pitch predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(PitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == 'SAME'
                                       else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self, xs):
        """

        :param xs: [B, T, H]
        :return: [B, T, H]
        """
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, H)
        return xs


class EnergyPredictor(PitchPredictor):
    pass


class StatsPredictor(nn.Module):
    def __init__(self, c_in=256, c_out=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(c_in, c_in, 3, 2),
            nn.ReLU(), nn.BatchNorm1d(c_in)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(c_in, c_in, 3, 2),
            nn.ReLU(), nn.BatchNorm1d(c_in)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(c_in, c_in, 3, 1)
        )
        self.out = Linear(c_in, c_out, bias=True)

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return: [B, c_out]
        """
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.out(x.mean(-1))


def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur


class ConvEmbedding(nn.Module):
    def __init__(self, inp_size, hidden_size, kernel_size=9):
        super().__init__()
        self.inp_size = inp_size
        padding = kernel_size // 2
        self.conv = ConvNorm(inp_size, hidden_size, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        """

        :param x: [B, T]
        :return:
        """
        x = F.one_hot(x, self.inp_size).float()  # x: [B, T, N_dict]
        x = x.transpose(1, 2)
        x = self.conv(x)  # [T, B, H]
        x = x.transpose(1, 2)
        return x


class ConvEmbedding2(nn.Module):
    def __init__(self, inp_size, hidden_size, kernel_size=9):
        super().__init__()
        self.inp_size = inp_size
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(inp_size, hidden_size)
        self.conv1d = nn.Conv1d(hidden_size, hidden_size, self.kernel_size)

    def forward(self, x):
        """

        :param x: [B, T]
        :return:
        """
        B, T = x.shape
        x_shifts = []
        for i in range(-(self.kernel_size // 2), self.kernel_size // 2 + 1):
            x_shifts.append(x + i)
        assert len(x_shifts) == self.kernel_size, (len(x_shifts), self.kernel_size)
        x_shifts = torch.stack(x_shifts, -1).clamp(min=0, max=self.inp_size - 1)  # [B, T, k]
        x_shifts = self.embedding(x_shifts).transpose(-1, -2)  # [B, T, K, H] -> [B, T, H, K]
        x_shifts = x_shifts.reshape(-1, self.hidden_size, self.kernel_size)  # [B*T, H, K]
        x_shifts = self.conv1d(x_shifts).reshape(B, T, self.hidden_size)  # [B*T, H, 1]
        return x_shifts


class ConvEmbedding3(nn.Module):
    def __init__(self, inp_size, hidden_size, kernel_size=5):
        super().__init__()
        self.inp_size = inp_size
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(inp_size, hidden_size)
        self.register_buffer('weights', torch.FloatTensor([0.1, 0.2, 0.4, 0.2, 0.1]))

    def forward(self, x):
        """

        :param x: [B, T]
        :return:
        """
        x_shifts = []
        for i in range(-(self.kernel_size // 2), self.kernel_size // 2 + 1):
            x_shifts.append(x + i)
        assert len(x_shifts) == self.kernel_size, (len(x_shifts), self.kernel_size)
        x_shifts = torch.stack(x_shifts, -1).clamp(min=0, max=self.inp_size - 1)  # [B, T, k]
        x_shifts = self.embedding(x_shifts).transpose(-1, -2)  # [B, T, K, H] -> [B, T, H, K]
        x_shifts = (x_shifts * self.weights[None, None, None, :]).sum(-1)
        return x_shifts


class Conv1dWithMask(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, bias=True, w_init_gain='linear'):
        super(Conv1dWithMask, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)

    def forward(self, x, mask=None):
        """

        :param x: [B, H, T]
        :param mask: non pad mask, shape: [B, T, T],
                      e.g.: tensor([[[1., 1., 0., 0., 0., 0., 0., 0.],
                                     [1., 1., 0., 0., 0., 0., 0., 0.],
                                     [1., 1., 1., 1., 0., 0., 0., 0.],
                                     [1., 1., 1., 1., 0., 0., 0., 0.],
                                     [1., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1., 1.]], ...])
        :return: [B, H', T]
        """
        # kernel_size = self.kernel_size
        # x = F.pad(x, [kernel_size // 2, kernel_size // 2])
        # return self.conv(x)  # TODO: replace with nomask conv1d
        x = x.transpose(1, 2)
        kernel_size = self.kernel_size
        B, T, H = x.shape

        if mask is not None:
            mask_pad = F.pad(mask, [kernel_size // 2, kernel_size // 2])
            mask_pad_shift = torch.cat([mask_pad[:, :, :-1].reshape(B, -1), mask_pad[:, :, -1]], -1)
            mask_pad_shift = mask_pad_shift.view(B, T, -1)[:, :, :kernel_size]
            mask_pad_shift = mask_pad_shift.view(-1, 1, kernel_size)
        else:
            mask_pad_shift = 1
        x_pad = F.pad(x, [0, 0, kernel_size // 2, kernel_size // 2], value=0)  # [B, T+K-1, H]
        x_unfold = x_pad.unfold(1, kernel_size, 1)  # [B, T, H, K]
        x_unfold = x_unfold.reshape(-1, H, kernel_size)  # [B*T, H, K]
        x_conv = self.conv(x_unfold * mask_pad_shift)  # [B*T, H', 1]
        x_conv = x_conv.reshape(B, T, self.out_channels)  # [B, T, H']
        return x_conv.transpose(1, 2)


class FFTBlocks(nn.Module):
    def __init__(self, hidden_size, num_layers, ffn_kernel_size=9, dropout=None, num_heads=2,
                 use_pos_embed=True, use_last_norm=True, norm='ln', use_pos_embed_alpha=True):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout if dropout is not None else hparams['dropout']
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm
        if use_pos_embed:
            self.max_source_positions = DEFAULT_MAX_TARGET_POSITIONS
            self.padding_idx = 0
            self.pos_embed_alpha = nn.Parameter(torch.Tensor([1])) if use_pos_embed_alpha else 1
            self.embed_positions = SinusoidalPositionalEmbedding(
                embed_dim, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS,
            )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.hidden_size, self.dropout,
                                    kernel_size=ffn_kernel_size, num_heads=num_heads)
            for _ in range(self.num_layers)
        ])
        if self.use_last_norm:
            if norm == 'ln':
                self.layer_norm = nn.LayerNorm(embed_dim)
            elif norm == 'bn':
                self.layer_norm = BatchNorm1dTBC(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, x, padding_mask=None, attn_mask=None, return_hiddens=False):
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :return: [B, T, C] or [L, B, T, C]
        """
        padding_mask = x.abs().sum(-1).eq(0).data if padding_mask is None else padding_mask
        nonpadding_mask_TB = 1 - padding_mask.transpose(0, 1).float()[:, :, None]  # [T, B, 1]
        if self.use_pos_embed:
            positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
            x = x + positions
            x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=padding_mask, attn_mask=attn_mask) * nonpadding_mask_TB
            hiddens.append(x)
        if self.use_last_norm:
            x = self.layer_norm(x) * nonpadding_mask_TB
        if return_hiddens:
            x = torch.stack(hiddens, 0)  # [L, T, B, C]
            x = x.transpose(1, 2)  # [L, B, T, C]
        else:
            x = x.transpose(0, 1)  # [B, T, C]
        return x


class FastspeechEncoder(FFTBlocks):
    def __init__(self, embed_tokens, hidden_size=None, num_layers=None, kernel_size=None, num_heads=2):
        hidden_size = hparams['hidden_size'] if hidden_size is None else hidden_size
        kernel_size = hparams['enc_ffn_kernel_size'] if kernel_size is None else kernel_size
        num_layers = hparams['dec_layers'] if num_layers is None else num_layers
        super().__init__(hidden_size, num_layers, kernel_size, num_heads=num_heads,
                         use_pos_embed=False)  # use_pos_embed_alpha for compatibility
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(hidden_size)
        self.padding_idx = 0
        self.embed_positions = SinusoidalPositionalEmbedding(
            hidden_size, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS,
        )

    def forward(self, txt_tokens):
        """

        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [T x B x C]
        }
        """
        encoder_padding_mask = txt_tokens.eq(self.padding_idx).data
        x = self.forward_embedding(txt_tokens)  # [B, T, H]
        x = super(FastspeechEncoder, self).forward(x, encoder_padding_mask)
        return x

    def forward_embedding(self, txt_tokens):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        if hparams['use_pos_embed']:
            positions = self.embed_positions(txt_tokens)
            x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class FastspeechDecoder(FFTBlocks):
    def __init__(self, hidden_size=None, num_layers=None, kernel_size=None, num_heads=None):
        num_heads = hparams['num_heads'] if num_heads is None else num_heads
        hidden_size = hparams['hidden_size'] if hidden_size is None else hidden_size
        kernel_size = hparams['dec_ffn_kernel_size'] if kernel_size is None else kernel_size
        num_layers = hparams['dec_layers'] if num_layers is None else num_layers
        super().__init__(hidden_size, num_layers, kernel_size, num_heads=num_heads)



class RefLnEncoderLayer(nn.Module):
    def __init__(self, in_channels, filter_size, kernel_size, stride, use_ln=True):
        super(RefLnEncoderLayer, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, filter_size,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=kernel_size // 2, bias=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.use_ln = use_ln
        if use_ln:
            self.ln = torch.nn.LayerNorm(filter_size)
        self.do = torch.nn.Dropout(hparams['dropout'])

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.use_ln:
            x = x.permute(0, 2, 1)
            x = self.ln(x)
            x = x.permute(0, 2, 1)
        x = self.do(x)
        return x


class RefEncoder(nn.Module):
    def __init__(self, in_channels, ref_hidden_stride_kernel, out_channels=None, ref_norm_layer=None):
        super().__init__()
        self.ref_encoder_layers = nn.ModuleList()
        if ref_norm_layer is None:
            ref_norm_layer = hparams['ref_norm_layer']
        self.strides = []
        for filter_stride_kernel in ref_hidden_stride_kernel:
            filter_size, stride, kernel_size = filter_stride_kernel.split(",")
            filter_size = int(filter_size)
            if filter_size == 0:
                filter_size = hparams['hidden_size']
            stride = int(stride)
            kernel_size = int(kernel_size)
            self.strides.append(stride)
            if ref_norm_layer == 'none':
                self.ref_encoder_layers += [
                    RefLnEncoderLayer(in_channels, filter_size, kernel_size, stride, use_ln=False)
                ]
            elif ref_norm_layer == 'ln':
                self.ref_encoder_layers += [
                    RefLnEncoderLayer(in_channels, filter_size, kernel_size, stride)
                ]
            elif ref_norm_layer == 'bn':
                self.ref_encoder_layers += [nn.Sequential(
                    torch.nn.Conv1d(in_channels, filter_size,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=kernel_size // 2, bias=True),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.BatchNorm1d(filter_size),
                    torch.nn.Dropout(hparams['dropout'])
                )]
            elif ref_norm_layer == 'gn':
                self.ref_encoder_layers += [nn.Sequential(
                    torch.nn.Conv1d(in_channels, filter_size,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=kernel_size // 2, bias=True),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.GroupNorm(16, filter_size),
                    torch.nn.Dropout(hparams['dropout'])
                )]
            in_channels = filter_size
        if out_channels is None:
            out_channels = hparams['hidden_size']
        self.project_out_dim = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        """

        :param x: [B, T, C]
        :return: [B, C]
        """
        # [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)
        for stride, l in zip(self.strides, self.ref_encoder_layers):
            x = l(x)
        # [B, C, T] -> [B, T, C]
        x = x.permute(0, 2, 1).mean(1)
        x = self.project_out_dim(x)
        return x  # [B, C]


#################################
# pwg adversarial pitch predictor
#################################
class PitchAdvPredictor(nn.Module):
    def __init__(self, idim, n_layers=3, n_chans=256, odim=2, kernel_size=1, dropout_rate=0.0, slope=0.2):
        super().__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                ConvNorm(in_chans, n_chans, kernel_size, stride=1),
                nn.LeakyReLU(slope, inplace=True),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = Linear(n_chans, odim)

    def forward(self, xs):
        """

        :param xs: [B, T, H]
        :return: [B, T, H]
        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, H)
        return xs


class PitchAdvMelEncoder(PitchAdvPredictor):
    pass


class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        B, T, _ = x.shape
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)  # [B*T_mel, N_vq]
        indices = torch.argmin(distances.float(), dim=-1)  # [B*T_mel]
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return x_flat, quantized, indices

    def forward(self, x):
        """

        :param x: [B, T, D]
        :return: [B, T, D]
        """
        B, T, _ = x.shape
        M, D = self.embedding.size()

        x_flat, quantized, indices = self.encode(x)
        encodings = F.one_hot(indices, M).float()
        indices = indices.reshape(B, T)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, indices, perplexity


class VQVAEVarianceEncoder(nn.Module):
    def __init__(self, hidden_size, num_vq_codes=64):
        super(VQVAEVarianceEncoder, self).__init__()
        self.pitch_embed = Embedding(300, hidden_size, 0)
        self.pitch_ref_encoder = RefEncoder(
            hidden_size, hparams['pitch_enc_hidden_stride_kernel'],
            ref_norm_layer=hparams['ref_norm_layer'])
        self.dur_embed = Embedding(32, hidden_size)
        self.dur_ref_encoder = RefEncoder(
            hidden_size, hparams['dur_enc_hidden_stride_kernel'],
            ref_norm_layer=hparams['ref_norm_layer'])
        self.n_embed = num_vq_codes
        self.vqvae = VQEmbeddingEMA(self.n_embed, hidden_size, commitment_cost=hparams['lambda_commit'])

    def forward(self, x, pitch=None, durs=None):
        """

        :param x: [B, T, H]
        :param mels: [B, T, 80]
        :return: [B, 1, H]
        """
        if pitch is not None:
            h_mel = self.pitch_ref_encoder(self.pitch_embed(pitch))  # [B, H]
            h_dur = self.dur_ref_encoder(self.dur_embed(durs))  # [B, H]
            h_ref = h_mel + h_dur
            z, vq_loss, vq_tokens, ppl = self.vqvae(h_ref[:, None, :])
            vq_loss = vq_loss.mean()
        else:
            z = torch.randint_like(x[:, :1, 0], 0, self.n_embed).long()  # [B, 1]
            z = F.embedding(z, self.vqvae.embedding)  # [B, 1, H]
            vq_loss = 0
        return z, vq_loss


class BertEncoder(nn.Module):
    def __init__(self, num_chars, hidden_size, num_layers=4, kernel_size=3):
        super(BertEncoder, self).__init__()
        self.fft = FFTBlocks(hidden_size, num_layers, kernel_size)
        self.out_proj = nn.Linear(hidden_size, num_chars)

    def forward(self, x):
        padding_mask = (x.abs().sum(-1) == 0)
        x = self.fft(x, padding_mask)
        return self.out_proj(x)

# class BertEncoder(nn.Module):
#     def __init__(self, num_chars, embedding_dim, n_convolutions=3, kernel_size=5):
#         super(BertEncoder, self).__init__()
#         convolutions = []
#         for _ in range(n_convolutions):
#             conv_layer = nn.Sequential(
#                 ConvNorm(embedding_dim,
#                          embedding_dim,
#                          kernel_size=kernel_size, stride=1,
#                          padding=int((kernel_size - 1) / 2),
#                          dilation=1, w_init_gain='relu'),
#                 nn.BatchNorm1d(embedding_dim))
#             convolutions.append(conv_layer)
#         self.convolutions = nn.ModuleList(convolutions)
#
#         self.lstm = nn.LSTM(embedding_dim, int(embedding_dim / 2), 1,
#                             batch_first=True, bidirectional=True)
#         self.out_proj = nn.Linear(embedding_dim, num_chars)
#
#     def forward(self, x):
#         input_lengths = (x.abs().sum(-1) > 0).sum(-1)
#         input_lengths = input_lengths.cpu().numpy()
#         x = x.transpose(1, 2)  # [B, H, T]
#         for conv in self.convolutions:
#             x = F.dropout(F.relu(conv(x)), 0.5, self.training) + x
#         x = x.transpose(1, 2)  # [B, T, H]
#         x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
#         self.lstm.flatten_parameters()
#         outputs, _ = self.lstm(x)
#         outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
#         return self.out_proj(outputs)
