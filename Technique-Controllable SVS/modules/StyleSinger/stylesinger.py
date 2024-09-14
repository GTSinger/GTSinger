import torch
from modules.fastspeech.tts_modules import PitchPredictor
from modules.StyleSinger.lse import ProsodyAligner, LocalStyleAdaptor
from utils.pitch_utils import f0_to_coarse, denorm_f0
from modules.commons.common_layers import *
from utils.hparams import hparams
from modules.StyleSinger.umln import  DistributionUncertainty
from modules.fastspeech.fs2 import FastSpeech2
from modules.fastspeech.tts_modules import DEFAULT_MAX_SOURCE_POSITIONS
from modules.diff.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from modules.diff.net import DiffNet,  DDiffNet
from modules.diff.prodiff import ProDiffusion


def expand_states(h, mel2token):
    h = F.pad(h, [0, 0, 1, 0])
    mel2token_ = mel2token[..., None].repeat([1, 1, h.shape[-1]])
    h = torch.gather(h, 1, mel2token_)  # [B, T, H]
    return h

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

class TechEncoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.mix_emb = nn.Embedding(2, hidden_channels, padding_idx=0)
        self.falsetto_emb = nn.Embedding(2, hidden_channels, padding_idx=0)
        self.breathy_emb = nn.Embedding(2, hidden_channels, padding_idx=0)
        self.pharyngeal_emb = nn.Embedding(2, hidden_channels, padding_idx=0)
        self.glissando_emb = nn.Embedding(2, hidden_channels, padding_idx=0)
        self.vibrato_emb = nn.Embedding(2, hidden_channels, padding_idx=0)

        nn.init.normal_(self.mix_emb.weight, 0.0, hidden_channels ** -0.5)
        nn.init.normal_(self.falsetto_emb.weight, 0.0, hidden_channels ** -0.5)
        nn.init.normal_(self.breathy_emb.weight, 0.0, hidden_channels ** -0.5)
        nn.init.normal_(self.pharyngeal_emb.weight, 0.0, hidden_channels ** -0.5)
        nn.init.normal_(self.glissando_emb.weight, 0.0, hidden_channels ** -0.5)
        nn.init.normal_(self.vibrato_emb.weight, 0.0, hidden_channels ** -0.5)

    def forward(self, mix,falsetto,breathy,pharyngeal,glissando,vibrato):
        mix = self.mix_emb(mix) * math.sqrt(self.hidden_channels)
        falsetto = self.falsetto_emb(falsetto) * math.sqrt(self.hidden_channels)
        breathy = self.breathy_emb(breathy) * math.sqrt(self.hidden_channels)
        pharyngeal = self.pharyngeal_emb(pharyngeal) * math.sqrt(self.hidden_channels)
        glissando = self.glissando_emb(glissando) * math.sqrt(self.hidden_channels)
        vibrato = self.vibrato_emb(vibrato) * math.sqrt(self.hidden_channels)

        x=mix+falsetto+breathy+pharyngeal+glissando+vibrato

        return x

DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins'])
}

class StyleSinger(FastSpeech2):
    '''
    [StyleSinger (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29932/31629): Style Transfer for Out-of-Domain Singing Voice Synthesis.
    '''
    def __init__(self, dictionary, out_dims=None):
        super().__init__(dictionary, out_dims)

        # note encoder
        self.note_encoder = NoteEncoder(n_vocab=100, hidden_channels=self.hidden_size)

        # emotion embedding
        if hparams['emo']:
            self.emo_embed_proj = Linear(hparams['emo_size'], self.hidden_size, bias=True)

        # technique embedding
        if hparams['tech']:
            self.tech_encoder = TechEncoder(hidden_channels=self.hidden_size)

        # UMLN
        if hparams['umln']:
            self.norm=DistributionUncertainty(p=0.5, alpha=0.1, eps=1e-6, hidden_size=self.hidden_size)

        # build style extractor
        if hparams['style']:
            self.style_extractor = LocalStyleAdaptor(self.hidden_size, hparams['nRQ'], self.padding_idx)
            self.l1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.align = ProsodyAligner(num_layers=2)

        # build f0 predictor
        if hparams["f0_gen"] == "gmdiff":
            self.gm_diffnet_inpainte = DDiffNet(in_dims=1, num_classes=2)
            self.f0_gen_inpainte = GaussianMultinomialDiffusion(num_classes=2, denoise_fn=self.gm_diffnet_inpainte, num_timesteps=hparams["f0_timesteps"])
        elif hparams["f0_gen"] == "conv":
            self.pitch_inpainter_predictor = PitchPredictor(
                self.hidden_size, n_chans=self.hidden_size,
                n_layers=5, dropout_rate=0.1, odim=2,
                padding=hparams['ffn_padding'], kernel_size=hparams['predictor_kernel'])

        # build attention layer
        self.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        self.embed_positions = SinusoidalPositionalEmbedding(
            self.hidden_size, self.padding_idx,
            init_size=self.max_source_positions + self.padding_idx + 1,
        )

        # build decoder
        if hparams['decoder']=='diffsinger':
            cond_hs = 80
            if hparams.get('use_txt_cond', True):
                cond_hs = cond_hs + hparams['hidden_size']
            cond_hs = cond_hs + hparams['hidden_size']   # for spk embedding
            from modules.diff.shallow_diffusion_tts import DiffusionDecoder
            self.ln_proj = nn.Linear(cond_hs, hparams["hidden_size"])
            self.diffsinger = DiffusionDecoder(
                phone_encoder=None,
                out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
                timesteps=hparams['timesteps'],
                K_step=hparams['K_step'],
                loss_type=hparams['diff_loss_type'],
                spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
            )
        elif hparams['decoder']=='prodiff':
            self.diff_decoder = ProDiffusion(
                out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
                timesteps=hparams['timesteps'], time_scale=hparams['timescale'],
                loss_type=hparams['diff_loss_type'],
                spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
            )

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None, emo_embed=None, ref_mels=None, ref_f0=None,
                f0=None, uv=None, skip_decoder=False, global_steps=0, infer=False, note=None, note_dur=None, note_type=None,
                mix=None,falsetto=None,breathy=None,pharyngeal=None,glissando=None,vibrato=None,**kwargs):
        ret = {}
        print(vibrato)
        # print(txt_tokens.size(),spk_embed.size(),emo_embed.size())
        encoder_out = self.encoder(txt_tokens)  # [B, T, C]
        note_out = self.note_encoder(note, note_dur, note_type)
        encoder_out = encoder_out + note_out
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]

        # add spk/emo embed
        ret['spk_embed'] =spk_embed = self.spk_embed_proj(spk_embed)[:, None, :]
        if hparams['emo']:
            ret['emo_embed'] =emo_embed = self.emo_embed_proj(emo_embed)[:, None, :]
        # add dur
        dur_inp = (encoder_out + spk_embed)
        # add tech
        if hparams['tech']:
            tech=self.tech_encoder(mix,falsetto,breathy,pharyngeal,glissando,vibrato)        

        if hparams['emo']:
            dur_inp += emo_embed
        if hparams['tech']:
            dur_inp +=tech         
        dur_inp *= src_nonpadding
        mel2ph = self.add_dur(dur_inp, mel2ph, txt_tokens, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = self.expand_states(encoder_out, mel2ph)
        if hparams['umln']:
            if hparams['emo']:
                decoder_inp = self.norm(decoder_inp, spk_embed + emo_embed)
            else:
                decoder_inp = self.norm(decoder_inp, spk_embed)

        if hparams['tech']:
            ret['tech']=tech = expand_states(tech, mel2ph)

        # add style RQ
        if hparams['style']:
            ret['ref_f0'] = ref_f0
            ret['style']=style = self.get_style(decoder_inp, ref_mels, ret, infer, global_steps)

        # add pitch embed
        midi_notes = None
        if infer:
            midi_notes = expand_states(note[:, :, None], mel2ph)
        pitch_inp_domain_specific = (decoder_inp + spk_embed)
        if hparams['emo']:
            pitch_inp_domain_specific +=emo_embed
        if hparams['tech']:
            pitch_inp_domain_specific+=tech
        pitch_inp_domain_specific *= tgt_nonpadding
        predicted_pitch = self.inpaint_pitch( pitch_inp_domain_specific, f0, uv, mel2ph, ret, encoder_out, midi_notes=midi_notes)

        # decode
        decoder_inp = decoder_inp + spk_embed + predicted_pitch 
        if hparams['emo']:
            decoder_inp +=emo_embed
        if hparams['style']:
            decoder_inp +=style
        if hparams['tech']:
            decoder_inp+=tech
        ret['decoder_inp'] = decoder_inp = decoder_inp * tgt_nonpadding
        if skip_decoder:
            return ret
        # prodiff
        if hparams['decoder']=='prodiff':
            ret = self.diff_decoder(decoder_inp, ret, ref_mels, infer=infer)
        #diffsinger
        elif hparams['decoder']=='diffsinger':
            ret['mel_out'] = self.run_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)
            is_training = self.training
            ret['x_mask'] = tgt_nonpadding
            self.run_diffsinger(ref_mels, infer, is_training, ret)
        elif hparams['decoder']=='fft':
            ret['mel_out'] = self.run_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)
        return ret

    def get_style(self, encoder_out, ref_mels, ret, infer=False, global_steps=0):
        # get RQ style
        ref_f0=ret['ref_f0']
        if global_steps > hparams['rq_start'] or infer:
            style_embedding, loss = self.style_extractor(ref_mels,ref_f0, no_rq=False)
            ret['rq_loss'] = loss
        else:
            style_embedding = self.style_extractor(ref_mels,ref_f0, no_rq=True)

        # add positional embedding
        positions = self.embed_positions(style_embedding[:, :, 0])
        style_embedding = self.l1(torch.cat([style_embedding, positions], dim=-1))

        # style-to-content attention
        src_key_padding_mask = encoder_out[:, :, 0].eq(self.padding_idx).data
        style_key_padding_mask = style_embedding[:, :, 0].eq(self.padding_idx).data

        if global_steps < hparams['forcing']:
            output, guided_loss, attn_emo = self.align(encoder_out.transpose(0, 1), style_embedding.transpose(0, 1),
                                                   src_key_padding_mask, style_key_padding_mask, forcing=True)
        else:
            output, guided_loss, attn_emo = self.align(encoder_out.transpose(0, 1), style_embedding.transpose(0, 1),
                                                       src_key_padding_mask, style_key_padding_mask, forcing=False)
        
        ret['gloss'] = guided_loss
        return output.transpose(0, 1)

    def inpaint_pitch(self, pitch_inp_domain_specific, f0, uv, mel2ph, ret, encoder_out=None, **kwargs):
        if hparams['pitch_type'] == 'frame':
            pitch_padding = mel2ph == 0
        if hparams['predictor_grad'] != 1:
            pitch_inp_domain_specific = pitch_inp_domain_specific.detach() + hparams['predictor_grad'] * (pitch_inp_domain_specific - pitch_inp_domain_specific.detach())

        if hparams["f0_gen"] == "gmdiff":
            pitch_domain_specific = self.add_gmdiff_pitch(pitch_inp_domain_specific, f0, uv, mel2ph, ret, **kwargs)
        elif hparams["f0_gen"] == "conv":
            pitch_domain_specific = self.pitch_inpainter_predictor(pitch_inp_domain_specific)

        pitch_pred = pitch_domain_specific
        ret['pitch_pred'] = pitch_pred

        use_uv = hparams['pitch_type'] == 'frame' and hparams['use_uv']
        if f0 is None:
            f0 = pitch_pred[:, :, 0]  # [B, T]
            if use_uv:
                uv = pitch_pred[:, :, 1] > 0  # [B, T]

        f0_denorm = denorm_f0(f0, uv if use_uv else None, hparams, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        ret['f0_denorm'] = f0_denorm
        ret['f0_denorm_pred'] = denorm_f0(pitch_pred[:, :, 0], (pitch_pred[:, :, 1] > 0) if use_uv else None, hparams, pitch_padding=pitch_padding)
        if hparams['pitch_type'] == 'ph':
            pitch = torch.gather(F.pad(pitch, [1, 0]), 1, mel2ph)
            ret['f0_denorm'] = torch.gather(F.pad(ret['f0_denorm'], [1, 0]), 1, mel2ph)
            ret['f0_denorm_pred'] = torch.gather(F.pad(ret['f0_denorm_pred'], [1, 0]), 1, mel2ph)
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
                x=torch.clamp(x,None,x_max)
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
            
            pitch_pred = self.f0_gen_inpainte(decoder_inp.transpose(-1, -2), None, None, None, ret, infer, dyn_clip=[lower_norm_f0, upper_norm_f0]) # [lower_norm_f0, upper_norm_f0]
            f0 = pitch_pred[:, :, 0]
            uv = pitch_pred[:, :, 1]
            uv[midi_notes[:, 0, :] == 0] = 1
            f0 = minmax_denorm(f0)
            ret["gdiff2"] = 0.0
            ret["mdiff2"] = 0.0

        else:
            nonpadding = (mel2ph > 0).float()
            norm_f0 = minmax_norm(f0)

            ret["mdiff2"], ret["gdiff2"], ret["nll2"] = self.f0_gen_inpainte(decoder_inp.transpose(-1, -2), norm_f0.unsqueeze(dim=1), uv, nonpadding, ret, infer)

        f0=f0[:,:,None]
        uv=uv[:,:,None]
        return torch.cat([f0,uv],dim=2)
    
    def run_diffsinger(self, tgt_mels, infer, is_training, ret):
        x_recon = ret['mel_out']
        g = x_recon.detach()
        B, T, _ = g.shape
        if hparams.get('use_txt_cond', True):
            g = torch.cat([g, ret['decoder_inp']], -1)
        g_spk_embed = ret['spk_embed'].repeat(1, T, 1)
        g = torch.cat([g, g_spk_embed], dim=-1)
        g = self.ln_proj(g)
        if not infer:
            if is_training:
                self.train()
        self.diffsinger(g, tgt_mels, x_recon, ret, infer)
