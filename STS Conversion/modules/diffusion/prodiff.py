import math
import random
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from einops import rearrange
from inspect import isfunction

from modules.fastspeech.fs2 import FastSpeech2
from utils.hparams import hparams

def vpsde_beta_t(t, T, min_beta, max_beta):
    t_coef = (2 * t - 1) / (T ** 2)
    return 1. - np.exp(-min_beta / T - 0.5 * (max_beta - min_beta) * t_coef)

def _logsnr_schedule_cosine(t, *, logsnr_min, logsnr_max):
    b = np.arctan(np.exp(-0.5 * logsnr_max))
    a = np.arctan(np.exp(-0.5 * logsnr_min)) - b
    return -2. * np.log(np.tan(a * t + b))

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def get_noise_schedule_list(schedule_mode, timesteps, min_beta=0.0, max_beta=0.01, s=0.008):
    if schedule_mode == "linear":
        schedule_list = np.linspace(0.000001, 0.01, timesteps)
    elif schedule_mode == "cosine":
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        schedule_list = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule_mode == "vpsde":
        schedule_list = np.array([
            vpsde_beta_t(t, timesteps, min_beta, max_beta) for t in range(1, timesteps + 1)])
    elif schedule_mode == "logsnr":
        u = np.array([t for t in range(0, timesteps + 1)])
        schedule_list = np.array([
            _logsnr_schedule_cosine(t / timesteps, logsnr_min=-20.0, logsnr_max=20.0) for t in range(1, timesteps + 1)])
    else:
        raise NotImplementedError
    return schedule_list

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

class ProDiffusion(nn.Module):
    def __init__(self, out_dims, denoise_fn,
                 timesteps=1000, time_scale=1, loss_type='l1', betas=None, spec_min=None, spec_max=None):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.mel_bins = out_dims

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = get_noise_schedule_list(
                schedule_mode=hparams['schedule_type'],
                timesteps=timesteps + 1,
                min_beta=0.1,
                max_beta=40,
                s=0.008,
            )

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.time_scale = time_scale
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('timesteps', to_torch(self.num_timesteps))      # beta
        self.register_buffer('timescale', to_torch(self.time_scale))      # beta
        self.register_buffer('betas', to_torch(betas))      # beta
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod)) # alphacum_t
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev)) # alphacum_{t-1}

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.register_buffer('spec_min', torch.FloatTensor(spec_min)[None, None, :hparams['keep_bins']])
        self.register_buffer('spec_max', torch.FloatTensor(spec_max)[None, None, :hparams['keep_bins']])

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_posterior_sample(self, x_start, x_t, t, repeat_noise=False):
        b, *_, device = *x_start.shape, x_start.device
        model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        noise = noise_like(x_start.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_start.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample(self, x_t, t, cond, spk_emb=None, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x_t.shape, x_t.device
        x_0_pred = self.denoise_fn(x_t, t, cond)

        return self.q_posterior_sample(x_start=x_0_pred, x_t=x_t, t=t)

    @torch.no_grad()
    def interpolate(self, x1, x2, t, cond, spk_emb, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        x = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc="interpolation sample time step", total=t):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond, spk_emb)
        x = x[:, 0].transpose(1, 2)
        return self.denorm_spec(x)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def diffuse_trace(self, x_start, mask):
        b, *_, device = *x_start.shape, x_start.device
        trace = [self.norm_spec(x_start).clamp_(-1., 1.) * ~mask.unsqueeze(-1)]
        for t in range(self.num_timesteps):
            t = torch.full((b,), t, device=device, dtype=torch.long)
            trace.append(
                self.diffuse_fn(x_start, t)[:, 0].transpose(1, 2) * ~mask.unsqueeze(-1)
            )
        return trace

    def diffuse_fn(self, x_start, t, noise=None):
        x_start = self.norm_spec(x_start)
        x_start = x_start.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
        zero_idx = t < 0 # for items where t is -1
        t[zero_idx] = 0
        noise = default(noise, lambda: torch.randn_like(x_start))
        out = self.q_sample(x_start=x_start, t=t, noise=noise)
        out[zero_idx] = x_start[zero_idx] # set x_{-1} as the gt mel
        return out

    def forward(self, cond, ret=None, ref_mels=None, infer=False):
        # cond [B, T, C]
        b, *_, device = *cond.shape, cond.device
        ret = {} if ret is None else ret
        if 'mel2ph' in ret:
            nonpadding = (ret['mel2ph'] != 0).float().unsqueeze(1).unsqueeze(1)  # [B, T] -> [B, 1, 1, T]
        else:
            nonpadding = (1 - cond.abs().sum(-1).eq(0).float().data).unsqueeze(1).unsqueeze(1)
        # cond = ret['decoder_inp'].transpose(1, 2)
        cond = cond.transpose(1, 2)
        if not infer:
            t = torch.randint(0, self.num_timesteps + 1, (b,), device=device).long()
            # Diffusion
            x_t = self.diffuse_fn(ref_mels, t) * nonpadding

            # Predict x_{start}
            x_0_pred = self.denoise_fn(x_t, t, cond) * nonpadding

            ret['mel_out'] = x_0_pred[:, 0].transpose(1, 2) # [B, T, 80]
        else:
            t = self.num_timesteps  # reverse总步数
            shape = (cond.shape[0], 1, self.mel_bins, cond.shape[2])
            x = torch.randn(shape, device=device)  # noise
            for i in tqdm(reversed(range(0, t)), desc='ProDiff Teacher sample time step', total=t):
                x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)  # x(mel), t, condition(phoneme)
            x = x[:, 0].transpose(1, 2)
            ret['mel_out'] = self.denorm_spec(x)  # 去除norm
        return ret

    def norm_spec(self, x):
        return x

    def denorm_spec(self, x):
        return x

    def cwt2f0_norm(self, cwt_spec, mean, std, mel2ph):
        return self.fs2.cwt2f0_norm(cwt_spec, mean, std, mel2ph)

    def out2mel(self, x):
        return x