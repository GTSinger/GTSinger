import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hparams import hparams
from utils.pitch_utils import f0_to_coarse, denorm_f0
from modules.commons.common_layers import LinearNorm, Linear, Embedding
from modules.speech2singing.modules import CrossFormerBlocks
from modules.diffusion.shallow_diffusion_tts import DiffusionDecoder
from modules.diffusion.diffusion import GaussianDiffusion as DDIM
from modules.diffusion.prodiff import ProDiffusion
from modules.fastspeech.tts_modules import FastspeechDecoder, LayerNorm, VQEmbeddingEMA


class ContentEncoder(nn.Module):
    def __init__(self, inp_size, hidden_size, n_layers=3, kernel_size=5, stride=1, padding=2, dilation=1):
        super(ContentEncoder, self).__init__()

        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.chs_grp = hparams['gn_chs_grp']

        convolutions = []
        for i in range(n_layers):
            conv_layer = nn.Sequential(
                nn.Conv1d(self.inp_size if i == 0 else self.hidden_size,
                          self.hidden_size, kernel_size, stride, padding, dilation),
                nn.GroupNorm(self.hidden_size // self.chs_grp, self.hidden_size))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

    def forward(self, x):
        x = x.transpose(1, 2)  # x [B, T, M] -> [B, M, T]
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        return x


class F0Encoder(nn.Module):
    def __init__(self, inp_size, hidden_size, use_pitch_embed, n_layers,  kernel_size=5, stride=1, padding=2, dilation=1):
        super(F0Encoder, self).__init__()

        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.chs_grp = hparams['gn_chs_grp']

        if use_pitch_embed:
            self.pitch_embed = Embedding(300, self.inp_size)

        convolutions = []
        for i in range(n_layers):
            conv_layer = nn.Sequential(
                nn.Conv1d(self.inp_size if i == 0 else self.hidden_size,
                          self.hidden_size, kernel_size, stride, padding, dilation),
                nn.GroupNorm(self.hidden_size // self.chs_grp, self.hidden_size))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

    def forward(self, x):
        # x: denorm_f0 [B, T]
        x = f0_to_coarse(x)
        x = self.pitch_embed(x)     # [B, T] -> [B, T, P]
        x = x.transpose(1, 2)       # x [B, T, P] -> [B, P, T]
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        return x


class RhythmPredictor(nn.Module):
    def __init__(self, hidden_size, out_size=8, cf_layers=1, num_guided_layers=0, num_heads=1, ffn_kernel=9,
                 dropout=0.1, cf_dropout=0.0, pred_kernel=3, pred_padding='SAME', pred_layers=2, pred_stride=1,
                 use_cont=True, attn_class='base', win_size=0.0, cb_config=None):
        super(RhythmPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.ffn_kernel = ffn_kernel
        self.dropout = dropout
        self.out_size = out_size
        self.num_guided_layers = num_guided_layers
        self.use_cont = use_cont

        if self.use_cont:
            self.crossformer = CrossFormerBlocks(
                hidden_size=self.hidden_size,
                num_layers=cf_layers,
                kv_size=self.hidden_size,
                ffn_kernel_size=self.ffn_kernel,
                num_heads=num_heads,
                num_guided_layers=num_guided_layers,
                dropout=cf_dropout,
                attn_class=attn_class,
                win_size=win_size,
                cb_config=cb_config
            )

        self.conv = torch.nn.ModuleList()
        self.kernel_size = pred_kernel
        self.padding = pred_padding
        for idx in range(pred_layers):
            in_chans = self.hidden_size
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2)
                                       if self.padding == 'SAME'
                                       else (self.kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, self.hidden_size, self.kernel_size, stride=pred_stride, padding=0),
                torch.nn.ReLU(),
                LayerNorm(self.hidden_size, dim=1),
                torch.nn.Dropout(self.dropout)
            )]

        self.linear = torch.nn.Linear(self.hidden_size, self.out_size)

    def forward(self, x, f0, spk_embed):
        # x [B, T2, H] or None
        # f0 [B, T1, H]
        # spk_embed [B, H]
        if self.use_cont:
            x, aw, gloss_spenv = self.crossformer(f0, x, need_weights=self.num_guided_layers > 0)     # [B, T, H]
        else:
            x, gloss_spenv = f0, 0
        x = x + spk_embed[:, None, :]
        x = x.transpose(1, 2)
        for f in self.conv:
            x = f(x)
        x = x.transpose(1, 2)       # [B, T, H]
        x = self.linear(x)
        return x, gloss_spenv


class DiffVAE2(nn.Module):
    def __init__(self, phone_encoder, denoise_fn_cls):
        super(DiffVAE2, self).__init__()

        self.hidden_size = hparams['diffvae_hidden']
        self.num_layers = hparams['diffvae_layers']
        self.ffn_kernel = hparams['diffvae_ffn_kernel']
        self.num_heads = hparams['diffvae_num_heads']
        self.dropout = hparams['diffvae_dropout']
        self.out_dims = hparams['audio_num_mel_bins']
        self.chs_grp = hparams['gn_chs_grp']

        self.content_encoder = ContentEncoder(inp_size=hparams.get('content_enc_inp', 80),
                                              hidden_size=self.hidden_size,
                                              n_layers=hparams['content_enc_layers'],
                                              kernel_size=hparams['content_enc_kernel'],
                                              stride=hparams['content_enc_stride'],
                                              padding=hparams['content_enc_padding'],
                                              dilation=hparams['content_enc_dilation'])
        self.f0_encoder = F0Encoder(inp_size=hparams['f0_embed_size'],
                                    hidden_size=self.hidden_size,
                                    use_pitch_embed=hparams['use_pitch_embed'],
                                    n_layers=hparams['f0_enc_layers'],
                                    kernel_size=hparams['f0_enc_kernel'],
                                    stride=hparams['f0_enc_stride'],
                                    padding=hparams['f0_enc_padding'],
                                    dilation=hparams['f0_enc_dilation'])

        self.rhythm_encoder = ContentEncoder(inp_size=hparams['audio_num_mel_bins'] if
                                             not hparams.get("use_sing_spenv_reduced", False) else 1,
                                             hidden_size=self.hidden_size,
                                             n_layers=hparams['rhythm_enc_layers'],
                                             kernel_size=hparams['rhythm_enc_kernel'],
                                             stride=hparams['rhythm_enc_stride'],
                                             padding=hparams['rhythm_enc_padding'],
                                             dilation=hparams['rhythm_enc_dilation'])

        if (hparams['use_sing_spenv'] or hparams.get('use_rhythm_pred', True)) and not hparams.get('wo_rhythm_adaptor'):
            self.rhythm_predictor = RhythmPredictor(hidden_size=self.hidden_size,
                                                    out_size=hparams['rhythm_n_embed'],
                                                    cf_layers=hparams.get('rhythm_pred_cf_layers', 1),
                                                    num_guided_layers=hparams.get('num_guided_layers', 0),
                                                    num_heads=hparams.get('rhythm_pred_num_heads', 1),
                                                    ffn_kernel=hparams['rhythm_pred_ffn_kernel'],
                                                    dropout=hparams['rhythm_pred_dropout'],
                                                    cf_dropout=hparams.get('rhythm_pred_cf_dropout', 0.0),
                                                    pred_kernel=hparams['rhythm_pred_kernel'],
                                                    pred_padding=hparams['rhythm_pred_padding'],
                                                    pred_layers=hparams['rhythm_pred_layers'],
                                                    pred_stride=hparams.get('rhythm_pred_stride', 1),
                                                    use_cont=hparams.get('rhythm_pred_use_cont', True),
                                                    attn_class=hparams.get('rhythm_pred_cf_type', 'base'),
                                                    win_size=hparams.get('rhythm_pred_win_size', 0.0),
                                                    cb_config={
                                                        'use_conv': hparams.get('cf_cb_use_conv', False),
                                                        'kernel_num': hparams.get('cf_cb_conv_kernel_num', 32),
                                                        'kernel_size': hparams.get('cf_cb_conv_kernel_size', 9),
                                                        'win_size': hparams.get('cf_cb_conv_win_size', 0.4),
                                                        'use_forward_attn': hparams.get('cf_cb_use_forward_attn', False)
                                                    })

            if hparams['rhythm_enc_stride'] > 1:
                assert hparams.get('rhythm_pred_stride', 1) > 1
                assert hparams['rhythm_enc_stride'] ** hparams['rhythm_enc_layers'] == \
                       hparams.get('rhythm_pred_stride', 1) ** hparams['rhythm_pred_layers']

            self.rhythm_vqvae = VQEmbeddingEMA(n_embeddings=hparams['rhythm_n_embed'],
                                               embedding_dim=self.hidden_size,
                                               commitment_cost=hparams['lambda_rhythm_commit'])

        if not hparams.get('wo_content_rhythm_aligner', False):
            self.crossformer = CrossFormerBlocks(
                hidden_size=self.f0_encoder.hidden_size,    # query
                num_layers=self.num_layers,
                kv_size=self.content_encoder.hidden_size,   # key and value
                ffn_kernel_size=self.ffn_kernel,
                num_heads=self.num_heads,
                dropout=self.dropout,
                num_guided_layers=hparams.get('num_guided_layers', 0),
                guided_sigma=hparams.get('guided_sigma', 0.3),
                attn_class=hparams.get('cf_type', 'base'),
                win_size=hparams.get('diffvae_win_size', 0.0),
                cb_config={
                    'use_conv': hparams.get('cf_cb_use_conv', False),
                    'kernel_num': hparams.get('cf_cb_conv_kernel_num', 32),
                    'kernel_size': hparams.get('cf_cb_conv_kernel_size', 9),
                    'win_size': hparams.get('cf_cb_conv_win_size', 0.4),
                    'use_forward_attn': hparams.get('cf_cb_use_forward_attn', False)
                }
            )

        if hparams.get('diffvae_decoder', 'diffsinger') in ['diffsinger', 'fs2']:
            self.fs2_decoder = FastspeechDecoder(
                num_heads=self.num_heads,
                hidden_size=hparams['hidden_size'],
                kernel_size=self.ffn_kernel,
                num_layers=hparams['fs2_layers']
            )
            self.mel_out = Linear(self.hidden_size, self.out_dims, bias=True)

        if hparams.get('diffvae_decoder', 'diffsinger') == 'diffsinger':
            denoise_fn = denoise_fn_cls()
            self.diff_decoder = DiffusionDecoder(
                phone_encoder=phone_encoder,
                out_dims=hparams['audio_num_mel_bins'],
                denoise_fn=denoise_fn,
                timesteps=hparams['timesteps'],
                K_step=hparams['K_step'],
                loss_type=hparams['diff_loss_type'],
                spec_min=hparams['spec_min'],
                spec_max=hparams['spec_max']
            )
        elif hparams.get('diffvae_decoder', 'diffsinger') == 'ddim':
            denoise_fn = denoise_fn_cls()
            self.diff_decoder = DDIM(out_dims=hparams['audio_num_mel_bins'],
                                     denoise_fn=denoise_fn,
                                     timesteps=hparams['timesteps'],
                                     loss_type=hparams['diff_loss_type'],
                                     spec_min=hparams['spec_min'],
                                     spec_max=hparams['spec_max'])
        elif hparams.get('diffvae_decoder', 'diffsinger') == 'prodiff':
            denoise_fn = denoise_fn_cls()
            self.diff_decoder = ProDiffusion(
                out_dims=hparams['audio_num_mel_bins'], denoise_fn=denoise_fn,
                timesteps=hparams['timesteps'], time_scale=hparams['timescale'],
                loss_type=hparams['diff_loss_type'],
                spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
            )

    def forward(self, mels, f0, spenv=None, spk_embed=None, ref_mels=None, ref_spenv=None, txt_tokens=None,
                mel2ph=None, uv=None, energy=None, infer=False, forcing=False, prev_tokens=None, **kwargs):
        # mels [B, T2, M] from source (speech)
        # f0 [B, T1]       from target (sing)
        # spenv [B, T2, M]
        # spk_embed [B, S]

        ret = {}
        if spk_embed is None:
            spk_embed = 0

        x_mel = self.content_encoder(mels)      # [B, T2, H]
        x_f0 = self.f0_encoder(f0)                  # [B, T1, H]

        x, aw_main, gloss_main = self.crossformer(
            x_f0, x_mel, need_weights=hparams.get('plot_attn_diffvae', False), forcing=forcing)  # [B, T1, H], T1 != T2
        ret['gloss_main'] = gloss_main

        if hparams.get('plot_attn_diffvae', False):
            ret['diffvae_aw_main'] = aw_main    # [B, T1, T2]

        r_code_pred, gloss_spenv = self.rhythm_predictor(x_mel, x_f0, spk_embed)  # [B, T1, 32] without softmax
        ret['gloss_spenv'] = gloss_spenv

        # quantize sing spenv
        if ref_spenv is not None:
            ref_spenv = self.rhythm_encoder(ref_spenv)
            q_rhythm, rq_loss, r_indices, perplexity = self.rhythm_vqvae(ref_spenv)
            ret['rq_loss'] = rq_loss
            ret['r_loss'] = F.cross_entropy(r_code_pred.transpose(1, 2), r_indices)  # pred [B, 32, T1], indices [B, T1]
        else:
            r_indices = torch.argmax(F.softmax(r_code_pred, dim=-1), dim=-1)       # [B, T1]
            q_rhythm = F.embedding(r_indices, self.rhythm_vqvae.embedding)  # [B, T1, H]
        # upsample if applicable
        if hparams['rhythm_enc_stride'] > 1 and hparams.get('rhythm_pred_stride', 1) > 1:
            scale = hparams['rhythm_enc_stride'] * hparams['rhythm_enc_layers']
            q_rhythm = F.upsample(q_rhythm.transpose(1, 2), scale_factor=scale).transpose(1, 2)[:, :f0.shape[1], :]  # upsample and cut
            r_indices = F.upsample(r_indices.float()[:, None, :], scale_factor=scale)[:, 0, :f0.shape[1]]  # not support 2D
            r_code_pred = F.upsample(r_code_pred.transpose(1, 2), scale_factor=scale).transpose(1, 2)[:, :f0.shape[1], :]
        ret['r_code_gt_pred'] = (r_indices, torch.argmax(F.softmax(r_code_pred, dim=-1), dim=-1))  # for visual
        ret['q_rhythm'] = q_rhythm  # add quantized rhythm visual

        encoder_out = x + spk_embed[:, None, :]
        if hparams.get('f0_bridge', True):
            encoder_out += x_f0
        if hparams.get('q_rhythm_bridge', True):
            encoder_out += q_rhythm

        ret['decoder_inp'] = decoder_inp = encoder_out

        if hparams.get('diffvae_decoder', 'diffsinger') == 'diffsinger':
            ret['mel_out'] = self.mel_out(self.fs2_decoder(decoder_inp))
            # diffusion decoder
            ret = self.diff_decoder(ret, txt_tokens, None, spk_embed, ref_mels, f0, uv, energy, infer, **kwargs)
        elif hparams.get('diffvae_decoder', 'diffsinger') == 'fs2':
            ret['mel_out'] = self.mel_out(self.fs2_decoder(decoder_inp))
        elif hparams.get('diffvae_decoder', 'diffsinger') == 'ddim':
            ret = self.diff_decoder(decoder_inp, ret, ref_mels, infer=infer)
        elif hparams.get('diffvae_decoder', 'diffsinger') == 'prodiff':
            ret = self.diff_decoder(decoder_inp, ret, ref_mels, infer=infer)

        return ret

    def out2mel(self, out):
        return out


class DiffVAE3(DiffVAE2):
    def __init__(self, phone_encoder, denoise_fn):
        super(DiffVAE3, self).__init__(phone_encoder, denoise_fn)

    def forward(self, mels, f0, spenv=None, spk_embed=None, ref_mels=None, ref_spenv=None, txt_tokens=None,
                mel2ph=None, uv=None, energy=None, infer=False, forcing=False, prev_tokens=None, **kwargs):
        # mels [B, T2, M] from source (speech)
        # f0 [B, T1]       from target (sing)
        # spenv [B, T2, M]
        # spk_embed [B, S]

        ret = {}
        if spk_embed is None:
            spk_embed = 0

        x_mel = self.content_encoder(mels)      # [B, T2, H]
        x_f0 = self.f0_encoder(f0)                  # [B, T1, H]

        if not hparams.get('wo_rhythm_adaptor', False):
            # predict rhythm
            r_code_pred, gloss_spenv = self.rhythm_predictor(x_mel, x_f0, spk_embed)  # [B, T1, 32] without softmax
            ret['gloss_spenv'] = gloss_spenv
            # quantize sing spenv
            if ref_spenv is not None:
                ref_spenv = self.rhythm_encoder(ref_spenv)
                q_rhythm, rq_loss, r_indices, perplexity = self.rhythm_vqvae(ref_spenv)
                ret['rq_loss'] = rq_loss
                ret['r_loss'] = F.cross_entropy(r_code_pred.transpose(1, 2), r_indices)  # pred [B, 32, T1], indices [B, T1]
            else:
                r_indices = torch.argmax(F.softmax(r_code_pred, dim=-1), dim=-1)  # [B, T1]
                q_rhythm = F.embedding(r_indices, self.rhythm_vqvae.embedding)  # [B, T1, H]
            # upsample if applicable
            if hparams['rhythm_enc_stride'] > 1 and hparams.get('rhythm_pred_stride', 1) > 1:
                scale = hparams['rhythm_enc_stride'] ** hparams['rhythm_enc_layers']
                q_rhythm = F.interpolate(q_rhythm.transpose(1, 2), scale_factor=scale).transpose(1, 2)[:, :f0.shape[1], :]  # upsample and cut
                r_indices = F.interpolate(r_indices.float()[:, None, :], scale_factor=scale)[:, 0, :f0.shape[1]]   # not support 2D
                r_code_pred = F.interpolate(r_code_pred.transpose(1, 2), scale_factor=scale).transpose(1, 2)[:, :f0.shape[1], :]

            ret['r_code_gt_pred'] = (r_indices, torch.argmax(F.softmax(r_code_pred, dim=-1), dim=-1))  # for visual
            ret['q_rhythm'] = q_rhythm  # add quantized rhythm visual
        else:
            spenv = self.rhythm_encoder(spenv)
            q_rhythm = F.interpolate(spenv.transpose(1, 2), size=x_f0.size(1)).transpose(1, 2)

        if not hparams.get('wo_content_rhythm_aligner', False):
            x, aw_main, gloss_main = self.crossformer(
                q_rhythm, x_mel, need_weights=hparams.get('plot_attn_diffvae', False), forcing=forcing)  # [B, T1, H], T1 != T2
            ret['gloss_main'] = gloss_main
            if hparams.get('plot_attn_diffvae', False):
                ret['diffvae_aw_main'] = aw_main    # [B, T1, T2]
        else:
            x = q_rhythm + F.interpolate(x_mel.transpose(1, 2), size=q_rhythm.size(1)).transpose(1, 2)

        non_padding_mask = (1 - x.abs().sum(-1).eq(0).float().data).unsqueeze(-1)   # [B, T, 1]

        # encoder_out = torch.cat((x, q_rhythm, x_f0, spk_embed.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        encoder_out = x + spk_embed[:, None, :]
        if hparams.get('f0_bridge', True):
            encoder_out += x_f0
        if hparams.get('q_rhythm_bridge', True):
            encoder_out += q_rhythm
        encoder_out = encoder_out * non_padding_mask

        ret['decoder_inp'] = decoder_inp = encoder_out

        if hparams.get('diffvae_decoder', 'diffsinger') == 'diffsinger':
            ret['mel_out'] = self.mel_out(self.fs2_decoder(decoder_inp))
            # diffusion decoder
            ret = self.diff_decoder(ret, txt_tokens, None, spk_embed, ref_mels, f0, uv, energy, infer, **kwargs)
        elif hparams.get('diffvae_decoder', 'diffsinger') == 'fs2':
            ret['mel_out'] = self.mel_out(self.fs2_decoder(decoder_inp))
        elif hparams.get('diffvae_decoder', 'diffsinger') == 'ddim':
            ret = self.diff_decoder(decoder_inp, ret, ref_mels, infer=infer)
        elif hparams.get('diffvae_decoder', 'diffsinger') == 'prodiff':
            ret = self.diff_decoder(decoder_inp, ret, ref_mels, infer=infer)

        return ret

    def out2mel(self, out):
        return out
