import math
import random
from collections import deque
from functools import partial
from inspect import isfunction
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from einops import rearrange
from utils.commons.hparams import hparams

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

class GaussianDiffusion(nn.Module):
    def __init__(self, out_dims, denoise_fn,
                 timesteps=1000, loss_type=hparams.get('diff_loss_type', 'l1'), beta_schedule="linear",linear_start=1e-4,
                 linear_end=2e-2,cosine_s=8e-3,):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.mel_bins = out_dims

        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.K_step = self.num_timesteps
        self.loss_type = loss_type

        self.noise_list = deque(maxlen=4)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

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

    def p_mean_variance(self, x, t, cond, clip_denoised, midi_f0=None, optional_cond=None):
        noise_pred = self.denoise_fn(x, t, cond=cond, optional_cond=optional_cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)
        if clip_denoised:
            x_recon.clamp_(-1, 1)
        # x_recon = x_recon + (midi_f0 - x_recon) * 0.25
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    def guided_p_mean_variance(self, x, t, cond, clip_denoised, midi_f0, match_fn, optional_cond=None):
        noise_pred = self.denoise_fn(x, t, cond=cond, optional_cond=optional_cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)
        x_recon.clamp_(-1., 1.)
        # with torch.enable_grad():
        #     x_in = x.detach().requires_grad_(True)
        #     noise_pred = self.denoise_fn(x_in, t, cond=cond)
        #     x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)
        #     x_recon.clamp_(-1., 1.)
        #     crit = torch.nn.CosineSimilarity(dim=-1)
        #     selected = - 1.0 * crit(x_recon, midi_f0).mean()
        #     grad = torch.autograd.grad(selected.sum(), x_in)[0]
        #     grad = grad * 20.0
        #     alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        #     eps = noise_pred - (1 - alpha_bar).sqrt() * grad
        #     x_recon = self.predict_start_from_noise(x, t=t, noise=eps)
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        x_recon = x_recon + (1 - alpha_bar).sqrt() * (midi_f0 - x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
        

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False, midi_f0=None, match_fn=None, optional_cond=None):
        b, *_, device = *x.shape, x.device
        if match_fn is not None:
            model_mean, _, model_log_variance = self.guided_p_mean_variance(x=x, t=t, cond=cond, clip_denoised=clip_denoised, midi_f0=midi_f0, match_fn=match_fn, optional_cond=optional_cond)
        else:
            model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond, clip_denoised=clip_denoised, midi_f0=midi_f0, optional_cond=optional_cond)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_plms(self, x, t, interval, cond, clip_denoised=True, repeat_noise=False):
        """
        Use the PLMS method from [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778).
        """

        def get_x_pred(x, noise_t, t):
            a_t = extract(self.alphas_cumprod, t, x.shape)
            a_prev = extract(self.alphas_cumprod, torch.max(t-interval, torch.zeros_like(t)), x.shape)
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

            x_delta = (a_prev - a_t) * ((1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x - 1 / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
            x_pred = x + x_delta

            return x_pred

        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond=cond)

        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(x_pred, max(t-interval, 0), cond=cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]) / 12
        elif len(noise_list) >= 3:
            noise_pred_prime = (55 * noise_pred - 59 * noise_list[-1] + 37 * noise_list[-2] - 9 * noise_list[-3]) / 24

        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)

        return x_prev

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond, noise=None, nonpadding=None, optional_cond=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond, (1-nonpadding), optional_cond=optional_cond)

        if self.loss_type == 'l1':
            if nonpadding is not None:
                loss = ((noise - x_recon).abs() * nonpadding.unsqueeze(1).unsqueeze(1)).mean()
            else:
                # print('are you sure w/o nonpadding?')
                loss = (noise - x_recon).abs().mean()

        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss
    
    def forward(self, inps, nonpadding=None, tgt=None, infer=False, midi_f0=None, optional_cond=None):
        b, *_, device = *inps.shape, inps.device
        cond = inps.transpose(-1, -2)
        optional_cond = optional_cond.transpose(-1, -2)

        if not infer:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            x = tgt
            x = x.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            diff_loss = self.p_losses(x, t, cond, nonpadding=nonpadding, optional_cond=optional_cond)
            return diff_loss, None
        else:
            t = self.num_timesteps
            shape = (cond.shape[0], 1, self.mel_bins, cond.shape[2])
            x = torch.randn(shape, device=device)

            if hparams.get('pndm_speedup'):
                self.noise_list = deque(maxlen=4)
                iteration_interval = hparams['pndm_speedup']
                for i in tqdm(reversed(range(0, t, iteration_interval)), desc='sample time step',
                              total=t // iteration_interval):
                    x = self.p_sample_plms(x, torch.full((b,), i, device=device, dtype=torch.long), iteration_interval,
                                           cond)
            else:
                midi_f0 = midi_f0.transpose(1, 2)[:, None, :, :]
                if hparams["midi_f0_prior"]:
                    t = 200
                    x = self.q_sample(x_start=midi_f0, t=torch.tensor([t - 1], device=device).long())
                for i in tqdm(reversed(range(0, t)), desc='sample time step', total=t):
                    x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond, midi_f0=midi_f0, optional_cond=optional_cond, match_fn=1)
            x = x[:, 0].transpose(1, 2)
            return 0.0, x

# pred x0 instead of noise in diffusion training
class GaussianDiffusionx0(GaussianDiffusion):
    def p_losses(self, x_start, t, cond, noise=None, nonpadding=None, optional_cond=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond, (1-nonpadding), optional_cond=optional_cond)
        nonpadding = nonpadding.unsqueeze(1).unsqueeze(1)
        loss = (F.l1_loss(x_recon, x_start, reduction='none') * nonpadding).sum() / nonpadding.sum()
        return loss
    
    def p_mean_variance(self, x, t, cond, clip_denoised, midi_f0=None, optional_cond=None):
        x_recon = self.denoise_fn(x, t, cond=cond, optional_cond=optional_cond)
        if clip_denoised:
            x_recon.clamp_(-1, 1)
        # x_recon = x_recon + (midi_f0 - x_recon) * 0.25
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    def guided_p_mean_variance(self, x, t, cond, clip_denoised, midi_f0, match_fn, optional_cond=None):
        x_recon = self.denoise_fn(x, t, cond=cond, optional_cond=optional_cond)
        x_recon.clamp_(-1, 1)
        # if t[0] < 200:
            # x_recon = (x_recon + t / 200 * midi_f0) / (1 + t / 200)
        x_recon = x_recon + (midi_f0 - x_recon) * 0.005
            # x_recon = (x_recon / torch.norm(x_recon, dim=-1) + t / 50 * midi_f0 / torch.norm(midi_f0, dim=-1)) / (1 + t / 50) * torch.norm(x_recon, dim=-1)
        # if t[0] < 100:
        #     lower = midi_f0 - 1 / 12
        #     upper = midi_f0 + 1 / 12
        #     x_recon.clamp_(lower, upper)
        # x_recon.clamp_(lower, upper)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance