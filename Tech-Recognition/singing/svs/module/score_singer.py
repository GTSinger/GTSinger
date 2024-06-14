# 针对乐谱处理的解决方案
import math

import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import nn
from modules.commons.layers import LayerNorm
from modules.commons.layers import Embedding
from modules.commons.transformer import MultiheadAttention
from utils.commons.hparams import hparams
from modules.tts.fs import FastSpeech
from modules.tts.fs import FS_ENCODERS
from modules.commons.nar_tts_modules import LengthRegulator, PitchPredictor
from utils.audio.pitch_utils import f0_to_coarse, denorm_f0
from singing.svs.module.diff.diff_f0 import GaussianDiffusionF0
from singing.svs.module.diff.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from singing.svs.module.diff.multinomial_diffusion import MultinomialDiffusion
from singing.svs.module.diff.net import DiffNet, F0DiffNet, DDiffNet, MDiffNet

def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur

class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask

def group_hidden_by_segs(h, seg_ids, max_len):
    """

    :param h: [B, T, H]
    :param seg_ids: [B, T]
    :return: h_ph: [B, T_ph, H]
    """
    B, T, H = h.shape
    h_gby_segs = h.new_zeros([B, max_len + 1, H]).scatter_add_(1, seg_ids[:, :, None].repeat([1, 1, H]), h)
    all_ones = h.new_ones(h.shape[:2])
    cnt_gby_segs = h.new_zeros([B, max_len + 1]).scatter_add_(1, seg_ids, all_ones).contiguous()
    h_gby_segs = h_gby_segs[:, 1:]
    cnt_gby_segs = cnt_gby_segs[:, 1:]
    h_gby_segs = h_gby_segs / torch.clamp(cnt_gby_segs[:, :, None], min=1)
    return h_gby_segs
# 使用mixture of gaussion 去解决attention monotonic 和忽略的问题
class MixGaussion(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.pre = ConvReluNorm(hidden_channels, hidden_channels, hidden_channels,
                                    kernel_size=5, n_layers=2, p_dropout=0)
        self.length_regulator = LengthRegulator()
        self.gaussian = nn.Conv1d(hidden_channels, 2, kernel_size=1)
    def forward(self, x, x_mask, x2word, mel2word, word_len, ret, pre_word=None,aux_inp=None, postfix='', infer=None):
        res_x = x
        if aux_inp is not None:
            aux_inp = aux_inp.detach() + \
            0.1 * (aux_inp - aux_inp.detach())
            x = x + aux_inp
        x = x.transpose(-1, -2)
        x_mask = x_mask.transpose(-1, -2)
        x = self.pre(x, x_mask)
        x = self.gaussian(x)
        m, logs = torch.split(x, 1, dim=1)
        m = (F.relu(m) + 1) * x_mask
        m = m[:, 0, :]
        B, T_ph = x2word.shape
        dur = torch.zeros([B, word_len.max() + 1]).to(x2word.device).scatter_add(1, x2word, m)
        m_word = dur[:, 1:]
        ret[f'{postfix}dur'] = m_word
        if hparams["dur_scale"] == 'log':
            ret[f"{postfix}dur"] = (m_word + 1).log()
        if mel2word is None:
            # if hparams['dur_scale'] == 'log':
            #     m_word = m_word.exp() - 1
            m_word = torch.clamp(torch.round(m_word), min=0).long()
            src_padding = res_x.data.abs().sum(-1) == 0
            # infer 时 batch size = 1
            mel2word = self.length_regulator(m_word, word_len.unsqueeze(dim=0))[..., 0].detach()
            t_mel = mel2word.shape[1]
            t_mel = t_mel // hparams.get("frames_multiple", 1) * hparams.get("frames_multiple", 1)
            mel2word = mel2word[:, :t_mel]
        ret["mel2word"] = mel2word
        ret["tgt_nonpadding"] = y_mask = (mel2word > 0).float()[:, :, None]
        # logs = (F.relu(logs)) * x_mask
        dur_word = mel2ph_to_dur(mel2word, word_len).float()
        scale = dur_word / (m_word + 1e-4)
        scale = torch.gather(F.pad(scale, [1, 0]), 1, x2word)
        m = m + (scale - 1) * m.detach()
        cumsum = torch.cumsum(m, dim=1)
        center = (cumsum - m / 2)
        position = torch.arange(mel2word.shape[1], dtype=mel2word.dtype, device=mel2word.device).unsqueeze(0)
        diff = center[:, None, :] - position[:, :, None]
        yx_mask = self.build_word_mask(mel2word, x2word)
        yx_mask = yx_mask * y_mask
        yx_mask = yx_mask * x_mask
        logits = -(diff ** 2 / 10.0)
        # logits = -(diff ** 2 / logs.exp())
        # logits = - torch.abs(diff) / (10.0)
        mask_logits = logits - ((1 - yx_mask) * 1e9)
        # print(logits)
        weights = F.softmax(mask_logits, dim=2, dtype=torch.float32)
        # print(weights)
        attn = torch.bmm(weights, res_x)
        # scale_loss = F.l1_loss((m_word + 1).log(), (dur_word + 1).log(), reduction="mean")
        return attn, weights, ret, mel2word

    @staticmethod
    def build_word_mask(x2word, y2word):
        return (x2word[:, :, None] == y2word[:, None, :]).long()

class DurationPredictor(torch.nn.Module):
    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0, padding='SAME'):
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
        self.linear = nn.Sequential(torch.nn.Linear(n_chans, 1), nn.Softplus())

    def forward(self, xs, x_masks=None):
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
            if x_masks is not None:
                xs = xs * (1 - x_masks.float())[:, None, :]
        xs = self.linear(xs.transpose(1, -1))[:, :, 0] + 1  # [B, T, C]，我们这里的dur 定义的是frames，所以需要+1
        xs = xs * (1 - x_masks.float())  # (B, T, C)
        return xs

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """

        :param x: [B, T]
        :return: [B, T, H]
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, :, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# 处理note_tokens, note_durs and note types
class NoteEncoder(nn.Module):
    def __init__(self, n_vocab, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels, padding_idx=0)
        self.type_emb = nn.Embedding(5, hidden_channels, padding_idx=0)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)
        nn.init.normal_(self.type_emb.weight, 0.0, hidden_channels ** -0.5)
        self.dur_ln = nn.Linear(1, hidden_channels)

    def forward(self, note_tokens, note_durs, note_types):
        x = self.emb(note_tokens) * math.sqrt(self.hidden_channels)
        types = self.type_emb(note_types) * math.sqrt(self.hidden_channels)
        durs = self.dur_ln(note_durs.unsqueeze(dim=-1))
        x = x + durs + types
        return x

# 只处理文本输入、音符处理、时长预测
class WordScoreEncoder(nn.Module):
    def __init__(self, dictionary, out_dims=None):
        super().__init__()
        # configs
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.hidden_size = hparams['hidden_size']
        self.out_dims = out_dims
        if out_dims is None:
            self.out_dims = hparams['audio_num_mel_bins']
        # build linguistic encoder
        self.encoder_embed_tokens = self.build_embedding(self.dictionary, self.hidden_size)
        self.ph_encoder = FS_ENCODERS[hparams['encoder_type']](hparams, self.encoder_embed_tokens, self.dictionary)

        self.note_encoder = NoteEncoder(100, self.hidden_size)
        self.sin_pos = SinusoidalPosEmb(self.hidden_size)
        # word and ph attention
        self.sin_pos = SinusoidalPosEmb(self.hidden_size)
        self.enc_pos_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.attn = MultiheadAttention(self.hidden_size, 1, encoder_decoder_attention=True, bias=False)
        self.attn.enable_torch_version = False
        # note and ph attn for dur
        if hparams["note_model"] == "attn":
            self.enc_pos_proj1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.ls_attn = MultiheadAttention(self.hidden_size, 1, encoder_decoder_attention=True, bias=False)
            self.ls_attn.enable_torch_version = False
        if hparams["note_model"] == "gaussian":
            self.dec_query_proj1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.note_gaussian = MixGaussion(self.hidden_size)
        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            n_chans=128,
            n_layers=hparams['dur_predictor_layers'],
            dropout_rate=hparams['predictor_dropout'], padding=hparams['ffn_padding'],
            kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()

    def forward(self, txt_tokens, ph2word, word_len, mel2word=None, mel2ph=None,
                infer=False, tgt_mels=None, note_tokens=None, note_durs=None, note_types=None, note2words=None, spk_embed=None
                ):
        ret = {}
        if hparams["note_model"] == "attn":
            x, expand_fusion, tgt_nonpadding = self.run_text_encoder(txt_tokens, ph2word, word_len, mel2word, mel2ph, note_tokens, note_durs, note_types, note2words, spk_embed, ret)
        elif hparams["note_model"] == "gaussian":
            x, expand_fusion, tgt_nonpadding = self.gaussian_text_encoder(txt_tokens, ph2word, word_len, mel2word, mel2ph, note_tokens, note_durs, note_types, note2words, spk_embed, ret, infer=infer)
        x = x * tgt_nonpadding
        ret['x_mask'] = tgt_nonpadding
        ret['decoder_inp'] = x
        ret['expand_fusion'] = expand_fusion
        return ret

    def run_text_encoder(self, txt_tokens, ph2word, word_len, mel2word, mel2ph, note_tokens, note_durs, note_types, note2word, spk_embed, ret, **kwargs):
        word2word = torch.arange(word_len)[None, :].to(ph2word.device) + 1  # [B, T_mel, T_word]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        note_nonpadding = (note_durs > 0).float()[:, :, None]
        # ph encoder
        ret['ph_encoder_out'] = ph_encoder_out = self.ph_encoder(txt_tokens) * src_nonpadding
        # note encoder
        ret['note_encoder_out'] = note_encoder_out = self.note_encoder(note_tokens, note_durs, note_types) * note_nonpadding
        # word dur extraction
        word_out = group_hidden_by_segs(ph_encoder_out, ph2word, word_len)
        expanded_word_out = self.expand_states(word_out, note2word)
        dur_inp = (note_encoder_out + expanded_word_out + spk_embed) * note_nonpadding
        ret["mel2word"] = mel2word = self.add_dur(dur_inp, mel2word, ret, ph2word=note2word, word_len=word_len)
        tgt_nonpadding = (mel2word > 0).float()[:, :, None]
        # ph and word attn
        ph_encoder_inp = (ph_encoder_out + spk_embed) * src_nonpadding
        enc_pos = self.build_pos_embed(word2word, ph2word)  # [B, T_ph, H]
        dec_pos = self.build_pos_embed(word2word, mel2word)  # [B, T_mel, H]
        dec_word_mask = self.build_word_mask(mel2word, ph2word)  # [B, T_mel, T_ph]
        x, weight = self.attention(ph_encoder_inp, enc_pos, dec_pos, dec_word_mask)
        ret['attn'] = weight
        x = torch.bmm(weight, ph_encoder_out)
        # note and word attn
        note_encoder_inp = (note_encoder_out + spk_embed) * note_nonpadding
        enc_pos1 = self.build_pos_embed(word2word, note2word)  # [B, T_note, H]
        dec_word_mask1 = self.build_word_mask(mel2word, note2word)
        note_x, weight = self.ls_attention(note_encoder_inp, enc_pos1, dec_pos, dec_word_mask1)
        ret['ls_attn'] = weight
        note_x = torch.bmm(weight, note_encoder_out)
        return x, note_x, tgt_nonpadding

    def gaussian_text_encoder(self, txt_tokens, ph2word, word_len, mel2word, mel2ph, note_tokens, note_durs, note_types, note2word, spk_embed, ret, **kwargs):
        word2word = torch.arange(word_len)[None, :].to(ph2word.device) + 1  # [B, T_mel, T_word]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        note_nonpadding = (note_durs > 0).float()[:, :, None]
        # ph encoder
        ret['ph_encoder_out'] = ph_encoder_out = self.ph_encoder(txt_tokens) * src_nonpadding
        # note encoder
        ret['note_encoder_out'] = note_encoder_out = self.note_encoder(note_tokens, note_durs, note_types) * note_nonpadding
        # note attn modeling
        word_out = group_hidden_by_segs(ph_encoder_out, ph2word, word_len)
        expanded_word_out = self.expand_states(word_out, note2word)
        aux_dur_inp = (expanded_word_out + spk_embed) * note_nonpadding
        expanded_note, n_w, ret, mel2word = self.note_gaussian(note_encoder_out,        note_nonpadding, note2word, mel2word, word_len, ret, aux_inp=aux_dur_inp, pre_word=word_out.detach(), postfix="", infer=kwargs.get("infer"))
        tgt_nonpadding = (mel2word > 0).float()[:, :, None]
        ret["ls_attn"] = n_w
       # ph and word attn
        ph_encoder_inp = (ph_encoder_out + spk_embed) * src_nonpadding
        enc_pos = self.build_pos_embed(word2word, ph2word)  # [B, T_ph, H]
        dec_pos = self.build_pos_embed(word2word, mel2word)  # [B, T_mel, H]
        dec_word_mask = self.build_word_mask(mel2word, ph2word)  # [B, T_mel, T_ph]
        x, weight = self.attention(ph_encoder_inp, enc_pos, dec_pos, dec_word_mask)
        ret['attn'] = weight
        x = torch.bmm(weight, ph_encoder_out)
        return x, expanded_note, tgt_nonpadding


    def attention(self, ph_encoder_out, enc_pos, dec_pos, dec_word_mask, dec_out=None
    ):
        ph_kv = self.enc_pos_proj(torch.cat([ph_encoder_out, enc_pos], -1))
        dec_q = dec_pos
        if dec_out is not None:
            dec_q = self.dec_query_proj1(torch.cat([dec_out, dec_pos], -1))
        ph_kv, dec_q = ph_kv.transpose(0, 1), dec_q.transpose(0, 1)
        x, (weight, _) = self.attn(dec_q, ph_kv, ph_kv, attn_mask=(1 - dec_word_mask) * -1e9)
        x = x.transpose(0, 1)
        x = x
        return x, weight
    
    def ls_attention(self, note_encoder_out, enc_pos, dec_pos, dec_word_mask):
        ph_kv = self.enc_pos_proj1(torch.cat([note_encoder_out, enc_pos], -1))
        dec_q = dec_pos
        ph_kv, dec_q = ph_kv.transpose(0, 1), dec_q.transpose(0, 1)
        x, (weight, _) = self.attn(dec_q, ph_kv, ph_kv, attn_mask=(1 - dec_word_mask) * -1e9)
        x = x.transpose(0, 1)
        x = x
        return x, weight

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def build_pos_embed(self, word2word, x2word):
        x_pos = self.build_word_mask(word2word, x2word).float()  # [B, T_word, T_ph]
        x_pos = (x_pos.cumsum(-1) / x_pos.sum(-1).clamp(min=1)[..., None] * x_pos).sum(1)
        x_pos = self.sin_pos(x_pos.float())  # [B, T_ph, H]
        return x_pos

    def add_dur(self, dur_input, mel2word, ret, **kwargs):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = dur_input.data.abs().sum(-1) == 0
        dur_input = dur_input.detach() + hparams['predictor_grad'] * (dur_input - dur_input.detach())
        dur = self.dur_predictor(dur_input, src_padding)
        ph2word = kwargs['ph2word']
        word_len = kwargs['word_len']
        B, T_ph = ph2word.shape
        dur = torch.zeros([B, word_len.max() + 1]).to(ph2word.device).scatter_add(1, ph2word, dur)
        dur = dur[:, 1:]
        ret['dur'] = dur
        if mel2word is None:
            if hparams['dur_scale'] == 'log':
                dur = dur.exp() - 1
            dur = torch.clamp(torch.round(dur), min=0).long()
            mel2word = self.length_regulator(dur, (1 - src_padding.long()).sum(-1))[..., 0].detach()
        return mel2word
    
    def expand_states(self, h, mel2ph):
        h = F.pad(h, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, h.shape[-1]])
        h = torch.gather(h, 1, mel2ph_)  # [B, T, H]
        return h

    @staticmethod
    def build_word_mask(x2word, y2word):
        return (x2word[:, :, None] == y2word[:, None, :]).long()

# baseline fs2
class FsWordSinger(FastSpeech):
    def __init__(self, dictionary, out_dims=None):
        super().__init__(dictionary, out_dims)
        del self.encoder
        del self.encoder_embed_tokens
        del self.dur_predictor
        del self.length_regulator
        self.encoder = WordScoreEncoder(dictionary, out_dims)   
        if not hparams["two_stage"]:
            del self.pitch_predictor
            del self.pitch_embed

    def forward(self, txt_tokens, ph2word, word_len, mel2word=None, mel2ph=None, spk_embed=None, infer_spk_embed=None, f0=None,
                uv=None, tgt_mels=None, infer=False, note_tokens = None, note_durs = None, note_types = None, note2words = None, mel2notes=None
                , **kwargs):
        # add spk embed
        if hparams['use_spk_embed'] or hparams['use_spk_id']:
            spk_embed = self.spk_embed_proj(spk_embed)[:, None, :]
        else:
            spk_embed = 0
        ret = self.encoder(txt_tokens, ph2word, word_len, mel2word, mel2ph, infer, tgt_mels,
                           note_tokens, note_durs, note_types, note2words, spk_embed)  # [B, T, C]
        decoder_inp = ret["decoder_inp"]
        expand_fusion = ret["expand_fusion"]
        mel2word = ret["mel2word"]
        if mel2word is not None:
            tgt_nonpadding = (mel2word > 0).float()[:, :, None]
        else:
            tgt_nonpadding = torch.ones([decoder_inp.shape[0], decoder_inp.shape[1], 1]).to(decoder_inp.device)
        
        # add pitch and energy embed
        pitch_inp = (expand_fusion + spk_embed + decoder_inp) * tgt_nonpadding
        expanded_midi_note = torch.bmm(ret["ls_attn"], note_tokens.unsqueeze(dim=-1).float())
        if hparams["two_stage"]:
            decoder_inp = decoder_inp + self.add_pitch(pitch_inp, f0, uv, mel2word, ret, encoder_out=decoder_inp, midi_notes=expanded_midi_note)
        else:
            decoder_inp = (expand_fusion + decoder_inp) * tgt_nonpadding
        # decoder_inp = decoder_inp + self.add_pitch(pitch_inp, decoder_inp, f0, uv, mel2ph, ret)
        ret['x_mask'] = tgt_nonpadding
        ret['decoder_inp'] = decoder_inp = (decoder_inp + spk_embed) * tgt_nonpadding
        if kwargs.get("skip_decoder", False):
            return ret
        ret = self.run_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, tgt_mels=tgt_mels)
        return ret

    def add_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None, **kwargs):
        if hparams['pitch_type'] == 'frame':
            pitch_pred_inp = decoder_inp
            pitch_padding = mel2ph == 0
        else:
            pitch_pred_inp = encoder_out
            pitch_padding = encoder_out.abs().sum(-1) == 0
            uv = None
        if hparams['predictor_grad'] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + \
                             hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
        ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp)
        use_uv = hparams['pitch_type'] == 'frame' and hparams['use_uv']
        if f0 is None:
            f0 = pitch_pred[:, :, 0]
            if use_uv:
                uv = pitch_pred[:, :, 1] > 0
                midi_notes = kwargs.get("midi_notes").transpose(-1, -2)
                uv[midi_notes[:, 0, :] == 0] = 1
        f0_denorm = denorm_f0(f0, uv if use_uv else None, hparams, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        ret['f0_denorm'] = f0_denorm
        ret['f0_denorm_pred'] = denorm_f0(
            pitch_pred[:, :, 0], (pitch_pred[:, :, 1] > 0) if use_uv else None,
            hparams, pitch_padding=pitch_padding)
        # ret['f0_denorm_pred'] = denorm_f0(
        #     pitch_pred[:, :, 0], uv if use_uv else None,
        #     hparams, pitch_padding=pitch_padding)
        if hparams['pitch_type'] == 'ph':
            pitch = torch.gather(F.pad(pitch, [1, 0]), 1, mel2ph)
            ret['f0_denorm'] = torch.gather(F.pad(ret['f0_denorm'], [1, 0]), 1, mel2ph)
            ret['f0_denorm_pred'] = torch.gather(F.pad(ret['f0_denorm_pred'], [1, 0]), 1, mel2ph)
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

    def run_decoder(self, decoder_inp, tgt_nonpadding, ret, infer, **kwargs):
        x = decoder_inp  # [B, T, H]
        x = self.decoder(x)
        x = self.mel_out(x)
        ret["mel_out"] = x * tgt_nonpadding
        return ret

class F0GenSinger(FsWordSinger):
    def __init__(self, dictionary, out_dims=None):
        super(F0GenSinger, self).__init__(dictionary=dictionary)
        self.dictionary = dictionary
        self.hidden_size = hparams["hidden_size"]
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        del self.pitch_predictor
        if hparams["f0_gen"] == "diff":
            self.uv_predictor = PitchPredictor(
                self.hidden_size, n_chans=predictor_hidden,
                n_layers=2, dropout_rate=0.1, odim=1,
                padding=hparams['ffn_padding'], kernel_size=hparams['predictor_kernel'])
            self.pitch_flow_diffnet = F0DiffNet(in_dims=1)
            self.f0_gen = GaussianDiffusionF0(out_dims=1, denoise_fn=self.pitch_flow_diffnet, timesteps=hparams["f0_timesteps"])
        elif hparams["f0_gen"] == "gmdiff":
            self.gm_diffnet = DDiffNet(in_dims=1, num_classes=2)
            self.f0_gen = GaussianMultinomialDiffusion(num_classes=2, denoise_fn=self.gm_diffnet, num_timesteps=hparams["f0_timesteps"])
        elif hparams["f0_gen"] == "mdiff":
            self.f0_predictor = PitchPredictor(
                self.hidden_size, n_chans=predictor_hidden,
                n_layers=2, dropout_rate=0.1, odim=1,
                padding=hparams['ffn_padding'], kernel_size=hparams['predictor_kernel'])
            self.m_diffnet = MDiffNet(num_classes=2)
            self.uv_gen = MultinomialDiffusion(num_classes=2, denoise_fn=self.m_diffnet, num_timesteps=hparams["f0_timesteps"])

    def add_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None, **kwargs):
        pitch_pred_inp = decoder_inp
        pitch_padding = mel2ph == 0
        # if hparams['predictor_grad'] != 1:
        #     pitch_pred_inp = pitch_pred_inp.detach() + \
        #                      hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
        if hparams["f0_gen"] == "diff":
            return self.add_diff_pitch(pitch_pred_inp, f0, uv, mel2ph, ret, encoder_out, **kwargs)
        elif hparams["f0_gen"] == "gmdiff":
            return self.add_gmdiff_pitch(pitch_pred_inp, f0, uv, mel2ph, ret, encoder_out, **kwargs)
        elif hparams["f0_gen"] == "mdiff":
            return self.add_mdiff_pitch(pitch_pred_inp, f0, uv, mel2ph, ret, encoder_out, **kwargs)

    def add_diff_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None, **kwargs):
        pitch_padding = mel2ph == 0
        if f0 is None:
            infer = True
        else:
            infer = False
        ret["uv_pred"] = uv_pred = self.uv_predictor(decoder_inp)
        def minmax_norm(x, uv=None):
            x_min = 6
            x_max = 10
            if torch.any(x> x_max):
                raise ValueError("check minmax_norm!!")
            normed_x = (x - x_min) / (x_max - x_min) * 2 - 1
            if uv is not None:
                normed_x[uv > 0] = 0
            return normed_x

        def minmax_denorm(x, uv=None):
            x_min = 6
            x_max = 10
            denormed_x = (x + 1) / 2 * (x_max - x_min) + x_min
            if uv is not None:
                denormed_x[uv > 0] = 0
            return denormed_x
        if infer:
            uv = uv_pred[:, :, 0] > 0
            midi_notes = kwargs.get("midi_notes").transpose(-1, -2)
            uv[midi_notes[:, 0, :] == 0] = 1
            uv = uv
            lower_bound = midi_notes - 1
            upper_bound = midi_notes + 1
            upper_norm_f0 = minmax_norm((2 ** ((upper_bound-69)/12) * 440).log2())
            lower_norm_f0 = minmax_norm((2 ** ((lower_bound-69)/12) * 440).log2())
            upper_norm_f0[upper_norm_f0 < -1] = -1
            upper_norm_f0[upper_norm_f0 > 1] = 1
            lower_norm_f0[lower_norm_f0 < -1] = -1
            lower_norm_f0[lower_norm_f0 > 1] = 1
            f0 = self.f0_gen(decoder_inp.transpose(-1, -2), None, None, ret, infer, dyn_clip=[lower_norm_f0, upper_norm_f0]) # 
            # f0 = self.f0_gen(decoder_inp.transpose(-1, -2), None, None, ret, infer)
            f0 = f0[:, :, 0]
            f0 = minmax_denorm(f0)
            ret["fdiff"] = 0.0
        else:
            # nonpadding = (mel2ph > 0).float() * (uv == 0).float()
            nonpadding = (mel2ph > 0).float()
            norm_f0 = minmax_norm(f0)
            ret["fdiff"] = self.f0_gen(decoder_inp.transpose(-1, -2), norm_f0, nonpadding.unsqueeze(dim=1), ret, infer)
        f0_denorm = denorm_f0(f0, uv, hparams, pitch_padding=pitch_padding)
        ret['f0_denorm_pred'] = f0_denorm
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

    def add_mdiff_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None, **kwargs):
        pitch_padding = mel2ph == 0
        if f0 is None:
            infer = True
        else:
            infer = False
        ret["f0_pred"] = f0_pred = self.f0_predictor(decoder_inp)
        if infer:
            # uv = uv
            f0 = f0_pred[:, :, 0]
            midi_notes = kwargs.get("midi_notes").transpose(-1, -2)
            uv = self.uv_gen(decoder_inp.transpose(-1, -2), None, None, ret, infer) # [lower_norm_f0, upper_norm_f0]
            uv = uv[:, :, 0]
            uv[midi_notes[:, 0, :] == 0] = 1
            # ret["fdiff"] = 0.0
        else:
            nonpadding = (mel2ph > 0).float()
            ret["mdiff"], ret["gdiff"], ret["nll"] = self.uv_gen(decoder_inp.transpose(-1, -2), uv, nonpadding, ret, infer)
        f0_denorm = denorm_f0(f0, uv, hparams, pitch_padding=pitch_padding)
        ret['f0_denorm_pred'] = f0_denorm
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

    def add_gmdiff_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None, **kwargs):
        pitch_padding = mel2ph == 0
        if f0 is None:
            infer = True
        else:
            infer = False
        def minmax_norm(x, uv=None):
            x_min = 6
            x_max = 10
            if torch.any(x> x_max):
                raise ValueError("check minmax_norm!!")
            normed_x = (x - x_min) / (x_max - x_min) * 2 - 1
            if uv is not None:
                normed_x[uv > 0] = 0
            return normed_x

        def minmax_denorm(x, uv=None):
            x_min = 6
            x_max = 10
            denormed_x = (x + 1) / 2 * (x_max - x_min) + x_min
            if uv is not None:
                denormed_x[uv > 0] = 0
            return denormed_x
        if infer:
            # uv = uv
            midi_notes = kwargs.get("midi_notes").transpose(-1, -2)
            lower_bound = midi_notes - 3 # 1 for good gtdur F0RMSE
            upper_bound = midi_notes + 3 # 1 for good gtdur F0RMSE
            upper_norm_f0 = minmax_norm((2 ** ((upper_bound-69)/12) * 440).log2())
            lower_norm_f0 = minmax_norm((2 ** ((lower_bound-69)/12) * 440).log2())
            upper_norm_f0[upper_norm_f0 < -1] = -1
            upper_norm_f0[upper_norm_f0 > 1] = 1
            lower_norm_f0[lower_norm_f0 < -1] = -1
            lower_norm_f0[lower_norm_f0 > 1] = 1
            pitch_pred = self.f0_gen(decoder_inp.transpose(-1, -2), None, None, None, ret, infer, dyn_clip=[lower_norm_f0, upper_norm_f0]) # [lower_norm_f0, upper_norm_f0]
            f0 = pitch_pred[:, :, 0]
            uv = pitch_pred[:, :, 1]
            uv[midi_notes[:, 0, :] == 0] = 1
            f0 = minmax_denorm(f0)
            # ret["fdiff"] = 0.0
        else:
            nonpadding = (mel2ph > 0).float()
            norm_f0 = minmax_norm(f0)
            ret["mdiff"], ret["gdiff"], ret["nll"] = self.f0_gen(decoder_inp.transpose(-1, -2), norm_f0.unsqueeze(dim=1), uv, nonpadding, ret, infer)
        f0_denorm = denorm_f0(f0, uv, hparams, pitch_padding=pitch_padding)
        ret['f0_denorm_pred'] = f0_denorm
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

# flow postnet
class FlowPostnet(nn.Module):
    def __init__(self):
        super().__init__()
        from modules.commons.normalizing_flow.glow_modules import Glow
        self.spk_embed_proj = Embedding(hparams['num_spk'], 80)
        cond_hs = 80
        cond_hs = 80 + 80 # spk embed
        self.post_flow = Glow(
            80, hparams['post_glow_hidden'], hparams['post_glow_kernel_size'], 1,
            hparams['post_glow_n_blocks'], hparams['post_glow_n_block_layers'],
            n_split=4, n_sqz=2,
            gin_channels=cond_hs,
            share_cond_layers=hparams['post_share_cond_layers'],
            share_wn_layers=hparams['share_wn_layers'],
            sigmoid_scale=hparams['sigmoid_scale']
        )
        self.prior_dist = dist.Normal(0, 1)
    
    def forward(self, tgt_mels, infer, ret, spk_embed):
        x_recon = ret['mel_out'].transpose(1, 2)
        g = x_recon.detach()
        spk_embed = self.spk_embed_proj(spk_embed)[:, :, None]
        B, _, T = g.shape
        spk_embed = spk_embed.repeat(1, 1, T)
        g = torch.cat([g, spk_embed], dim=1)
        prior_dist = self.prior_dist
        if not infer:
            x_mask = ret['x_mask'].transpose(1, 2)
            y_lengths = x_mask.sum(-1)
            tgt_mels = tgt_mels.transpose(1, 2)
            z_postflow, ldj = self.post_flow(tgt_mels, x_mask, g=g)
            ldj = ldj / y_lengths / 80
            ret['z_pf'], ret['ldj_pf'] = z_postflow, ldj
            ret['postflow'] = -prior_dist.log_prob(z_postflow).mean() - ldj.mean()
        else:
            x_mask = torch.ones_like(x_recon[:, :1, :])
            z_post = prior_dist.sample(x_recon.shape).to(g.device) * hparams['noise_scale']
            x_recon_, _ = self.post_flow(z_post, x_mask, g, reverse=True)
            x_recon = x_recon_
            ret['mel_out'] = x_recon.transpose(1, 2)
            ret["postflow"] = 0.0

DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins'])
}

class DiffPostnet(nn.Module):
    def __init__(self):
        super().__init__()
        from singing.svs.module.diff.shallow_diffusion_tts import GaussianDiffusionPostnet
        self.spk_embed_proj = Embedding(hparams['num_spk'], 80)
        self.ln_proj = nn.Linear(160, hparams["hidden_size"])
        self.postdiff = GaussianDiffusionPostnet(
            phone_encoder=None,
            out_dims=80, denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
    
    def forward(self, tgt_mels, infer, ret, spk_embed):
        x_recon = ret['mel_out']
        g = x_recon.detach()
        spk_embed = self.spk_embed_proj(spk_embed)[:, None, :]
        B, T, _ = g.shape
        spk_embed = spk_embed.repeat(1, T, 1)
        g = torch.cat([g, spk_embed], dim=-1)
        g = self.ln_proj(g)
        self.postdiff(g, tgt_mels, x_recon, ret, infer)
        