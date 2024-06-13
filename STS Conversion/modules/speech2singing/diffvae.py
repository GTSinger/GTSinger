import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hparams import hparams
from utils.pitch_utils import f0_to_coarse, denorm_f0
from modules.commons.common_layers import LinearNorm, Linear, Embedding
from modules.speech2singing.modules import CrossFormerBlocks
from modules.diffusion.shallow_diffusion_tts import DiffusionDecoder
from modules.fastspeech.tts_modules import FastspeechDecoder, LayerNorm

class RhythmEncoder(nn.Module):
    """Rhythm Encoder
    """

    def __init__(self, inp_size, hidden_size, neck_size, freq=1, n_layers=1,
                 kernel_size=5, stride=1, padding=2, dilation=1, rnn_layers=1):
        super().__init__()

        self.neck_size = neck_size
        self.freq = freq
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.chs_grp = hparams['gn_chs_grp']
        self.out_size = self.neck_size * 2

        convolutions = []
        for i in range(n_layers):
            conv_layer = nn.Sequential(
                nn.Conv1d(self.inp_size if i == 0 else self.hidden_size,
                          self.hidden_size, kernel_size, stride, padding, dilation),
                nn.GroupNorm(self.hidden_size // self.chs_grp, self.hidden_size))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(self.hidden_size, self.neck_size, rnn_layers, batch_first=True, bidirectional=True)

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)   # x [B, T, M] -> [B, M, T]
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        if mask is not None:
            outputs = outputs * mask
        out_forward = outputs[:, :, :self.neck_size]
        out_backward = outputs[:, :, self.neck_size:]

        codes = torch.cat((out_forward[:, self.freq - 1::self.freq, :], out_backward[:, ::self.freq, :]), dim=-1)

        return codes


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
    def __init__(self, f0_inp_size, rhythm_inp_size, spk_embed_size, hidden_size, out_size=80,
                 ffn_kernel=9, dropout=0.1, pred_kernel=3, pred_padding='SAME', pred_layers=2):
        super(RhythmPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.ffn_kernel = ffn_kernel
        self.dropout = dropout
        self.f0_inp_size = f0_inp_size
        self.out_size = out_size

        self.spk_embed_proj = Linear(spk_embed_size, self.hidden_size)
        if f0_inp_size != self.hidden_size:
            self.f0_embed_proj = Linear(f0_inp_size, self.hidden_size)
        else:
            self.f0_embed_proj = nn.Identity()
        if rhythm_inp_size != self.hidden_size:
            self.rhythm_embed_proj = Linear(rhythm_inp_size, self.hidden_size)
        else:
            self.rhythm_embed_proj = nn.Identity()

        self.crossformer = CrossFormerBlocks(
            hidden_size=self.hidden_size,
            num_layers=1,
            kv_size=self.hidden_size,
            ffn_kernel_size=self.ffn_kernel,
            num_heads=1
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
                torch.nn.Conv1d(in_chans, self.hidden_size, self.kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(self.hidden_size, dim=1),
                torch.nn.Dropout(self.dropout)
            )]

        self.linear = torch.nn.Linear(self.hidden_size, self.out_size)

    def forward(self, spenv, f0, spk_embed):
        spk_embed = self.spk_embed_proj(spk_embed)[:, None, :]
        f0 = self.f0_embed_proj(f0)
        spenv = self.rhythm_embed_proj(spenv)

        x = self.crossformer(f0, spenv)     # [B, T, H]
        x = x + spk_embed
        x = x.transpose(1, 2)
        for f in self.conv:
            x = f(x)
        x = x.transpose(1, 2)
        x = self.linear(x)
        return x

class DiffVAE(nn.Module):
    def __init__(self, phone_encoder, denoise_fn):
        super(DiffVAE, self).__init__()

        self.hidden_size = hparams['diffvae_hidden']
        self.num_layers = hparams['diffvae_layers']
        self.ffn_kernel = hparams['diffvae_ffn_kernel']
        self.num_heads = hparams['diffvae_num_heads']
        self.dropout = hparams['diffvae_dropout']
        self.out_dims = hparams['audio_num_mel_bins']
        self.chs_grp = hparams['gn_chs_grp']

        self.content_encoder = ContentEncoder(inp_size=hparams['audio_num_mel_bins'],
                                              hidden_size=hparams['content_enc_hidden'],
                                              n_layers=hparams['content_enc_layers'],
                                              kernel_size=hparams['content_enc_kernel'],
                                              stride=hparams['content_enc_stride'],
                                              padding=hparams['content_enc_padding'],
                                              dilation=hparams['content_enc_dilation'])
        self.f0_encoder = F0Encoder(inp_size=hparams['f0_embed_size'],
                                    hidden_size=hparams['f0_enc_hidden'],
                                    use_pitch_embed=hparams['use_pitch_embed'],
                                    n_layers=hparams['f0_enc_layers'],
                                    kernel_size=hparams['f0_enc_kernel'],
                                    stride=hparams['f0_enc_stride'],
                                    padding=hparams['f0_enc_padding'],
                                    dilation=hparams['f0_enc_dilation'])
        self.rhythm_encoder = RhythmEncoder(inp_size=hparams['audio_num_mel_bins'],
                                            hidden_size=hparams['rhythm_enc_hidden'],
                                            neck_size=hparams['rhythm_enc_neck'],
                                            freq=hparams['rhythm_enc_freq'],
                                            n_layers=hparams['rhythm_enc_layers'],
                                            kernel_size=hparams['rhythm_enc_kernel'],
                                            stride=hparams['rhythm_enc_stride'],
                                            padding=hparams['rhythm_enc_padding'],
                                            dilation=hparams['rhythm_enc_dilation'],
                                            rnn_layers=hparams['rhythm_enc_rnn_layers'])

        if hparams['use_sing_spenv']:
            self.rhythm_predictor = RhythmPredictor(f0_inp_size=hparams['f0_enc_hidden'],
                                                    rhythm_inp_size=self.rhythm_encoder.out_size,
                                                    spk_embed_size=hparams['spk_embed_size'],
                                                    hidden_size=hparams['rhythm_pred_hidden'],
                                                    out_size=hparams['audio_num_mel_bins'],
                                                    ffn_kernel=hparams['rhythm_pred_ffn_kernel'],
                                                    dropout=hparams['rhythm_pred_dropout'],
                                                    pred_kernel=hparams['rhythm_pred_kernel'],
                                                    pred_padding=hparams['rhythm_pred_padding'],
                                                    pred_layers=hparams['rhythm_pred_layers'])

        self.crossformer = CrossFormerBlocks(
            hidden_size=self.f0_encoder.hidden_size,    # query
            num_layers=self.num_layers,
            kv_size=self.content_encoder.hidden_size,   # key and value
            ffn_kernel_size=self.ffn_kernel,
            num_heads=self.num_heads
        )

        if hparams.get('diffvae_enc_proj_type', 'linear') == 'linear':
            self.enc_proj = nn.Sequential(
                LinearNorm(in_dim=self.crossformer.hidden_size + hparams['spk_embed_size'] +
                           (self.rhythm_predictor.out_size if hparams.get('use_sing_spenv', False)
                            else self.rhythm_encoder.out_size) + self.f0_encoder.hidden_size,
                           out_dim=self.hidden_size),
                nn.Dropout(self.dropout)
            )
        elif hparams.get('diffvae_enc_proj_type', 'linear') == 'conv':
            self.enc_proj = nn.Sequential(
                nn.Conv1d(self.crossformer.hidden_size + hparams['spk_embed_size'] +
                          (self.rhythm_predictor.out_size if hparams.get('use_sing_spenv', False)
                           else self.rhythm_encoder.out_size) + self.f0_encoder.hidden_size,
                          self.hidden_size, 5, 1, 2, 1),
                nn.ReLU(),
                nn.GroupNorm(self.hidden_size // self.chs_grp, self.hidden_size),
                nn.Dropout(self.dropout)
            )

        self.fs2_decoder = FastspeechDecoder(
            num_heads=self.num_heads,
            hidden_size=hparams['hidden_size'],
            kernel_size=self.ffn_kernel,
            num_layers=hparams['fs2_layers']
        )
        self.mel_out = Linear(self.hidden_size, self.out_dims, bias=True)

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

    def forward(self, mels, f0, spenv, spk_embed, ref_mels, ref_spenv=None, txt_tokens=None, mel2ph=None,
                uv=None, energy=None, infer=False, **kwargs):
        # mel_mono [B, T2, M] from source (speech)
        # f0 [B, T1, P]       from target (sing)
        # spenv [B, T2, M]
        # spk_embed [B, S]

        ret = {}

        x_mel = self.content_encoder(mels)      # [B, T2, H2]
        x_f0 = self.f0_encoder(f0)                  # [B, T1, H1]
        x_spenv = self.rhythm_encoder(spenv)        # [B, T2, H3]

        x = self.crossformer(x_f0, x_mel)           # [B, T1, H], T1 != T2

        if not hparams.get('use_sing_spenv', False):
            # spenv interpolate
            x_spenv = x_spenv.transpose(1, 2)           # [B, T2, H3] -> [B, H3, T2]
            x_spenv = F.interpolate(x_spenv, size=x.size(1))    # [B, H3, T2] -> [B, H3, T1]
            x_spenv = x_spenv.transpose(1, 2)           # [B, H3, T1] -> [B, T1, H3]
        else:
            x_spenv = self.rhythm_predictor(x_spenv, x_f0, spk_embed)
            ret['spenv_pred'] = x_spenv
            if ref_spenv is not None:
                x_spenv = ref_spenv                         # [B, T1, M]

        encoder_out = torch.cat((x, x_spenv, x_f0, spk_embed.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        if hparams.get('diffvae_enc_proj_type', 'linear') == 'linear':
            encoder_out = self.enc_proj(encoder_out)
        elif hparams.get('diffvae_enc_proj_type', 'linear') == 'conv':
            encoder_out = self.enc_proj(encoder_out.transpose(1, 2))
            encoder_out = encoder_out.transpose(1, 2)

        ret['decoder_inp'] = decoder_inp = encoder_out
        ret['mel_out'] = self.mel_out(self.fs2_decoder(decoder_inp))

        # diffusion decoder
        ret = self.diff_decoder(ret, txt_tokens, None, spk_embed, ref_mels, f0, uv, energy, infer, **kwargs)

        return ret

    def out2mel(self, out):
        return out
