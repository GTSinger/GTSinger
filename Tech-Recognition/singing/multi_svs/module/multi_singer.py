import torch.nn as nn
from modules.tts.fs2_orig import FastSpeech2Orig
import math
import torch
from utils.audio.pitch_utils import denorm_f0, f0_to_coarse
from modules.commons.nar_tts_modules import PitchPredictor
from modules.tts.commons.align_ops import expand_states
from singing.multi_svs.module.diffusion import GaussianDiffusion
from singing.multi_svs.module.diffnet import ComposerFFT
from utils.commons.hparams import hparams
from modules.commons.layers import Embedding
from modules.commons.transformer import MultiheadAttention
from modules.tts.fs import FS_ENCODERS
import torch.nn.functional as F

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
    
# multi-singer/speaker svs module, without consider note
class MultiSpeaker(FastSpeech2Orig):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__(dict_size, hparams, out_dims)
        self.ln_proj = nn.Linear(1, 256)
        self.scale_predictor = PitchPredictor(
                self.hidden_size, n_chans=128,
                n_layers=5, dropout_rate=0.1, odim=1,
                kernel_size=hparams['predictor_kernel'])
        
    def forward(self, txt_tokens, mel2ph=None, spk_embed=None,
                f0=None, uv=None, energy=None, infer=False, **kwargs):
        ret = {}
        encoder_out = self.encoder(txt_tokens)  # [B, T, C]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]

        # add dur
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = decoder_inp_ = expand_states(encoder_out, mel2ph)
        style_embed = self.spk_id_proj(spk_embed)[:, None, :]
        # add pitch and energy embed
        random_f0 = kwargs.get("random_f0")
        pitch_inp = (decoder_inp_ + style_embed) * tgt_nonpadding
        decoder_inp = decoder_inp + self.forward_pitch(pitch_inp, f0, random_f0, uv, mel2ph, ret, encoder_out)

        # add pitch and energy embed
        # energy_inp = (decoder_inp_ + style_embed) * tgt_nonpadding
        # decoder_inp = decoder_inp + self.forward_energy(energy_inp, energy, ret)

        # decoder input
        ret['decoder_inp'] = decoder_inp = (decoder_inp + style_embed) * tgt_nonpadding
        ret['mel_out'] = self.forward_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)
        return ret
    
    def forward_pitch(self, decoder_inp, f0, random_f0, uv, mel2ph, ret, encoder_out=None):
        pitch_pred_inp = decoder_inp
        pitch_padding = mel2ph == 0
        if self.hparams['predictor_grad'] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + \
                             self.hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
        scale_pred_inp = pitch_pred_inp + self.ln_proj(random_f0[:, :, None])
        scale_pred_inp = scale_pred_inp * (mel2ph > 0).float()[:, :, None]
        ret['f0_restore'] = self.scale_predictor(scale_pred_inp).mean(1) * random_f0
        # ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp)
        use_uv = self.hparams['pitch_type'] == 'frame' and self.hparams['use_uv']
        f0_denorm = denorm_f0(f0, uv if use_uv else None, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        ret['f0_denorm'] = f0_denorm
        ret['f0_denorm_pred'] = f0_denorm
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

    def forward_energy(self, decoder_inp, energy, ret):
        decoder_inp = decoder_inp.detach() + self.hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())
        # ret['energy_pred'] = energy_pred = self.energy_predictor(decoder_inp)[:, :, 0]
        # energy_embed_inp = energy_pred if energy is None else energy
        energy_embed_inp = energy
        energy_embed_inp = torch.clamp(energy_embed_inp * 256 // 4, min=0, max=255).long()
        energy_embed = self.energy_embed(energy_embed_inp)
        return energy_embed

class WordScoreEncoder(nn.Module):
    def __init__(self, dict_size, out_dims=None):
        super().__init__()
        # configs
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.hidden_size = hparams['hidden_size']
        self.out_dims = out_dims
        if out_dims is None:
            self.out_dims = hparams['audio_num_mel_bins']
        # build linguistic encoder
        self.ph_encoder = FS_ENCODERS[hparams['encoder_type']](hparams, dict_size)

        # word and ph attention
        self.sin_pos = SinusoidalPosEmb(self.hidden_size)
        self.enc_pos_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.attn = MultiheadAttention(self.hidden_size, 1, encoder_decoder_attention=True, bias=False)
        self.attn.enable_torch_version = False

    def forward(self, txt_tokens, ph2word, word_len, mel2word=None, mel2ph=None, spk_embed=None):
        ret = {}
        x, tgt_nonpadding = self.run_text_encoder(txt_tokens, ph2word, word_len, mel2word, mel2ph, spk_embed, ret)
        ret['x_mask'] = tgt_nonpadding
        ret['decoder_inp'] = x
        return ret

    def run_text_encoder(self, txt_tokens, ph2word, word_len, mel2word, mel2ph, spk_embed, ret, **kwargs):
        word2word = torch.arange(word_len)[None, :].to(ph2word.device) + 1  # [B, T_mel, T_word]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        # ph encoder
        ret['ph_encoder_out'] = ph_encoder_out = self.ph_encoder(txt_tokens) * src_nonpadding
        # word dur extraction
        tgt_nonpadding = (mel2word > 0).float()[:, :, None]
        # ph and word attn
        ph_encoder_inp = (ph_encoder_out + spk_embed) * src_nonpadding
        enc_pos = self.build_pos_embed(word2word, ph2word)  # [B, T_ph, H]
        dec_pos = self.build_pos_embed(word2word, mel2word)  # [B, T_mel, H]
        dec_word_mask = self.build_word_mask(mel2word, ph2word)  # [B, T_mel, T_ph]
        x, weight = self.attention(ph_encoder_inp, enc_pos, dec_pos, dec_word_mask)
        ret['attn'] = weight
        x = torch.bmm(weight, ph_encoder_out)
        return x, tgt_nonpadding

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

    def build_pos_embed(self, word2word, x2word):
        x_pos = self.build_word_mask(word2word, x2word).float()  # [B, T_word, T_ph]
        x_pos = (x_pos.cumsum(-1) / x_pos.sum(-1).clamp(min=1)[..., None] * x_pos).sum(1)
        x_pos = self.sin_pos(x_pos.float())  # [B, T_ph, H]
        return x_pos
    
    def expand_states(self, h, mel2ph):
        h = F.pad(h, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, h.shape[-1]])
        h = torch.gather(h, 1, mel2ph_)  # [B, T, H]
        return h

    @staticmethod
    def build_word_mask(x2word, y2word):
        return (x2word[:, :, None] == y2word[:, None, :]).long()

class MultiSinger(FastSpeech2Orig):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__(dict_size, hparams, out_dims)
        # del self.encoder
        # self.encoder = WordScoreEncoder(dict_size, out_dims)
        # denoise_fn = FFT(hparams['hidden_size'], hparams['diff_layers'], hparams['diff_kernel_size'], hparams['diff_heads'])
        # self.f0_gen = GaussianDiffusionx0(1, denoise_fn=denoise_fn)
        # predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        # self.uv_predictor = PitchPredictor(
        #         self.hidden_size, n_chans=predictor_hidden,
        #         n_layers=5, dropout_rate=0.1, odim=1,
        #         kernel_size=hparams['predictor_kernel'])

    def forward(self, txt_tokens, ph2word, word_len, mel2ph=None, mel2word=None, spk_embed=None, f0=None, uv=None, energy=None, infer=False, **kwargs):
        ret = {}
        encoder_out = self.encoder(txt_tokens)  # [B, T, C]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        style_embed = self.spk_embed_proj(spk_embed)[:, None, :]
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = decoder_inp_ = expand_states(encoder_out, mel2ph)

        # style_embed = self.spk_embed_proj(spk_embed)[:, None, :]
        # ret = self.encoder(txt_tokens, ph2word, word_len, mel2word,mel2ph, style_embed)  # [B, T, C]

        # tgt_nonpadding = ret["x_mask"]
        # decoder_inp_ = ret["decoder_inp"] * tgt_nonpadding
        # decoder_inp = decoder_inp_
        
        # add pitch 
        pitch_inp = (decoder_inp_ + style_embed) * tgt_nonpadding
        decoder_inp = decoder_inp + self.forward_pitch(pitch_inp, f0, uv, mel2ph, ret, None, infer, midi_f0=kwargs.get("midi_f0"))
        # decoder
        ret['decoder_inp'] = decoder_inp = (decoder_inp + style_embed) * tgt_nonpadding
        ret['mel_out'] = self.forward_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)
        return ret

    def forward_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None, infer=False, midi_f0=None):
        pitch_pred_inp = decoder_inp
        pitch_padding = mel2ph == 0
        f0_denorm = denorm_f0(f0, uv, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)
        ret['f0_denorm'] = f0_denorm
        # if self.hparams['predictor_grad'] != 1:
        #     pitch_pred_inp = pitch_pred_inp.detach() + \
        #                      self.hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
        # ret['uv_pred'] = uv_pred = self.uv_predictor(pitch_pred_inp)
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
            # uv = uv_pred[:, :, 0] > 0
            # midi_f0 = torch.log2(midi_f0 + 1e-8)
            # midi_f0 =midi_f0
            # diff_loss, f0_pred = self.f0_gen(pitch_pred_inp, infer=True, midi_f0=minmax_norm(midi_f0)[:, :, None])

            # midi_f0 = denorm_f0(midi_f0, uv, pitch_padding=pitch_padding)
            ret["midi_f0"] = midi_f0 = denorm_f0(midi_f0, uv, pitch_padding=pitch_padding)
            # f0_pred = minmax_denorm(f0_pred[:, :, 0])
            f0_pred = f0
            f0_denorm_pred = denorm_f0(f0_pred, uv, pitch_padding=pitch_padding)
            pitch = f0_to_coarse(f0_denorm_pred)
            ret['f0_denorm_pred'] = f0_denorm_pred
        else:
            f0_pred = f0
            nonpadding = (mel2ph > 0).float()
            # nonpadding = nonpadding * (uv == 0).float()
            # diff_loss, _ = self.f0_gen(pitch_pred_inp, nonpadding, tgt=minmax_norm(f0)[:, :, None])
        # ret["l_fd"] = diff_loss

        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

class PitchDiff(nn.Module):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__()
        self.p_singer = MultiSinger(dict_size, hparams, out_dims)
        self.uv_embed = Embedding(2, hparams["hidden_size"], 0)
        self.wm_f0_proj = nn.Linear(1, 64)
        self.sm_f0_proj = nn.Linear(1, 64)
        self.mid_f0_proj = nn.Linear(1, 64)
        self.mask_embed = Embedding(2, 64, 0)
        denoise_fn = ComposerFFT(hparams['diff_hidden_size'], hparams['diff_layers'], hparams['diff_kernel_size'], hparams['diff_heads'])
        self.f0_gen = GaussianDiffusion(1, denoise_fn=denoise_fn)
    
    def forward(self, txt_tokens, ph2word, word_len, mel2ph=None, mel2word=None, spk_embed=None, f0=None, uv=None, energy=None, infer=False, **kwargs):
        ret = {}
        encoder_out = self.p_singer.encoder(txt_tokens)  # [B, T, C]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        style_embed = self.p_singer.spk_embed_proj(spk_embed)[:, None, :]
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = decoder_inp_ = expand_states(encoder_out, mel2ph)
        
        # add pitch 
        pitch_inp = (decoder_inp_ + style_embed) * tgt_nonpadding
        uv_embedding = self.uv_embed(uv.long()) * tgt_nonpadding
        pitch_inp = (pitch_inp + uv_embedding) * tgt_nonpadding
        b, t, _ = pitch_inp.shape

        if kwargs.get("mid_f0") is not None:
            cond1 = self.mid_f0_proj(kwargs["mid_f0"][:, :, None]) * tgt_nonpadding
        else:
            cond1 = torch.zeros([b, t, 64]).to(pitch_inp.device)

        if kwargs.get("wm_f0") is not None:
            cond2 = self.wm_f0_proj(kwargs["wm_f0"][:, :, None]) * tgt_nonpadding
            cond2 = (cond2 + self.mask_embed(kwargs["wm"])) * tgt_nonpadding
            
        else:
            cond2 = torch.zeros([b, t, 64]).to(pitch_inp.device)

        if kwargs.get("sm_f0") is not None:
            cond3 = self.sm_f0_proj(kwargs["sm_f0"][:, :, None]) * tgt_nonpadding
            cond3 = (cond3 + self.mask_embed(kwargs["sm"])) * tgt_nonpadding
            
        else:
            cond3 = torch.zeros([b, t, 64]).to(pitch_inp.device)
        
        optional_cond = torch.cat([cond1, cond2, cond3], dim=-1)
        
        decoder_inp = decoder_inp + self.forward_pitch(pitch_inp, f0, uv, mel2ph, ret, None, infer, midi_f0=kwargs.get("midi_f0"), optional_cond=optional_cond)
        # decoder
        if infer:
            ret['decoder_inp'] = decoder_inp = (decoder_inp + style_embed) * tgt_nonpadding
            x = self.p_singer.decoder(decoder_inp)
            ret['mel_out'] = self.p_singer.mel_out(x) * tgt_nonpadding
        return ret
    
    def forward_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None, infer=False, midi_f0=None, optional_cond=None):
        pitch_pred_inp = decoder_inp
        pitch_padding = mel2ph == 0
        f0_denorm = denorm_f0(f0, uv, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)
        ret['f0_denorm'] = f0_denorm
        # ret['uv_pred'] = uv_pred = self.uv_predictor(pitch_pred_inp)
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
            diff_loss, f0_pred = self.f0_gen(pitch_pred_inp, infer=True, midi_f0=minmax_norm(midi_f0)[:, :, None], optional_cond=optional_cond)

            # midi_f0 = denorm_f0(midi_f0, uv, pitch_padding=pitch_padding)
            ret["midi_f0"] = midi_f0 = denorm_f0(midi_f0, uv, pitch_padding=pitch_padding)
            f0_pred = minmax_denorm(f0_pred[:, :, 0])
            f0_denorm_pred = denorm_f0(f0_pred, uv, pitch_padding=pitch_padding)
            pitch = f0_to_coarse(f0_denorm_pred)
            ret['f0_denorm_pred'] = f0_denorm_pred
        else:
            f0_pred = f0
            nonpadding = (mel2ph > 0).float()
            diff_loss, _ = self.f0_gen(pitch_pred_inp, nonpadding, tgt=minmax_norm(f0)[:, :, None], optional_cond=optional_cond)
        ret["l_fd"] = diff_loss
        pitch_embed = self.p_singer.pitch_embed(pitch)
        return pitch_embed
        

        
