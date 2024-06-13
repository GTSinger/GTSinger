# code from https://github.com/rotot0/tab-ddpm/blob/main/tab_ddpm/gaussian_multinomial_diffsuion.py

import torch.nn.functional as F
import torch
import math
from utils.hparams import hparams
import numpy as np
eps = 1e-8

import torch
import numpy as np
import torch.nn.functional as F
from torch.profiler import record_function
from inspect import isfunction
from tqdm import tqdm
from functools import partial

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

# 不同长度的序列需要mask
def sum_except_batch_with_padding(x, nonpadding_mask=None):
    # x [b, t]
    # padding_mask [b, t]
    x = x * nonpadding_mask
    sum_x = sum_except_batch(x)
    sum_nonpadding = sum_except_batch(nonpadding_mask)
    return sum_x / sum_nonpadding

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def exists(x):
    return x is not None

def extract(a, t, x_shape):
    b, *_ = t.shape
    t = t.to(a.device)
    out = a.gather(-1, t)
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

class FoundNANsError(BaseException):
    """Found NANs during sampling"""
    def __init__(self, message='Found NANs during sampling.'):
        super(FoundNANsError, self).__init__(message)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def linear_beta_schedule(timesteps, max_beta=hparams.get('max_beta', 0.01)):
    """
    linear schedule
    """
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas

class GaussianMultinomialDiffusion(torch.nn.Module):
    def __init__(
            self,
            num_classes,
            denoise_fn,
            num_timesteps=1000,
            gaussian_loss_type='mse',
            gaussian_parametrization='eps',
            multinomial_loss_type='vb_stochastic',
            parametrization='x0'
        ):

        super(GaussianMultinomialDiffusion, self).__init__()
        assert multinomial_loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        if multinomial_loss_type == 'vb_all':
            print('Computing the loss using the bound on _all_ timesteps.'
                  ' This is expensive both in terms of memory and computation.')

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.num_classes = num_classes 
        self._denoise_fn = denoise_fn
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.multinomial_loss_type = multinomial_loss_type
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization
        betas = linear_beta_schedule(num_timesteps, max_beta=hparams['f0_max_beta'])
        alphas = 1. - betas
        alphas = torch.tensor(alphas.astype('float64'))
        # multimodal diffusion
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('log_alpha', to_torch(log_alpha))
        self.register_buffer('log_1_min_alpha', to_torch(log_1_min_alpha))
        self.register_buffer('log_cumprod_alpha', to_torch(log_cumprod_alpha))
        self.register_buffer('log_1_min_cumprod_alpha', to_torch(log_1_min_cumprod_alpha))
        self.register_buffer('Lt_history', torch.zeros(num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(num_timesteps))

        # Gaussian diffusion
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

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

    # Gaussian part
    def gaussian_q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def gaussian_p_losses(self, x_recon, noise=None, nonpadding=None):
        loss = ((noise - x_recon).abs() * nonpadding.unsqueeze(dim=1)).sum() / (nonpadding.unsqueeze(dim=1) + 1e-8).sum()
        return loss

    def gaussian_predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def gaussian_q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def gaussian_p_mean_variance(self, x, t, noise_pred, clip_denoised=False, dyn_clip=None):
        x_recon = self.gaussian_predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            if dyn_clip is not None:
                x_recon.clamp_(dyn_clip[0], dyn_clip[1])
            else:
                x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.gaussian_q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def gaussian_p_sample(self, x, t, noise_pred, clip_denoised=True, repeat_noise=False, dyn_clip=None):
        # perform a x_t ==> x_t-1 step
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.gaussian_p_mean_variance(x=x, t=t, noise_pred=noise_pred, clip_denoised=clip_denoised, dyn_clip=dyn_clip)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # Multinomial part

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )

        return log_probs

    def predict_start(self, model_out, log_x_t, t, out_dict):

        # model_out = self._denoise_fn(x_t, t.to(x_t.device), **out_dict)

        assert model_out.size(0) == log_x_t.size(0)
        assert model_out.size(1) == self.num_classes, f'{model_out.size()}'

        log_pred = F.log_softmax(model_out, dim=1)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)


        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, model_out, log_x, t, out_dict):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(model_out, log_x, t=t, out_dict=out_dict)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(model_out, log_x, t=t, out_dict=out_dict)
        else:
            raise ValueError
        return log_model_pred

    @torch.no_grad()
    def p_sample(self, model_out, log_x, t, out_dict):
        model_log_prob = self.p_pred(model_out, log_x=log_x, t=t, out_dict=out_dict)
        out = self.log_sample_categorical(model_log_prob)
        return out

    @torch.no_grad()
    def p_sample_loop(self, shape, out_dict):
        device = self.log_alpha.device

        b = shape[0]
        # start with random normal image.
        img = torch.randn(shape, device=device)

        for i in reversed(range(1, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), out_dict)
        return img

    @torch.no_grad()
    def _sample(self, image_size, out_dict, batch_size = 16):
        return self.p_sample_loop((batch_size, 3, image_size, image_size), out_dict)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def nll(self, log_x_start, out_dict):
        b = log_x_start.size(0)
        device = log_x_start.device
        loss = 0
        for t in range(0, self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()

            kl = self.compute_Lt(
                log_x_start=log_x_start,
                log_x_t=self.q_sample(log_x_start=log_x_start, t=t_array),
                t=t_array,
                out_dict=out_dict)

            loss += kl

        loss += self.kl_prior(log_x_start)

        return loss

    def kl_prior(self, log_x_start, nonpadding=None):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch_with_padding(kl_prior, nonpadding)

    def compute_Lt(self, model_out, log_x_start, log_x_t, t, out_dict, detach_mean=False, nonpadding=None):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        log_model_prob = self.p_pred(model_out, log_x=log_x_t, t=t, out_dict=out_dict)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch_with_padding(kl, nonpadding_mask=nonpadding)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch_with_padding(decoder_nll, nonpadding)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = (Lt_sqrt / Lt_sqrt.sum()).to(device)

            t = torch.multinomial(pt_all, num_samples=b, replacement=True).to(device)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _multinomial_loss(self, model_out, log_x_start, log_x_t, t, pt, out_dict, nonpadding):

        if self.multinomial_loss_type == 'vb_stochastic':
            kl = self.compute_Lt(
                model_out, log_x_start, log_x_t, t, out_dict, 
            nonpadding=nonpadding)
            kl_prior = self.kl_prior(log_x_start, nonpadding=nonpadding)
            # Upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior
            return vb_loss

        elif self.multinomial_loss_type == 'vb_all':
            # Expensive, dont do it ;).
            # DEPRECATED
            return -self.nll(log_x_start)
        else:
            raise ValueError()

    def log_prob(self, x, out_dict):
        b, device = x.size(0), x.device
        if self.training:
            return self._multinomial_loss(x, out_dict)

        else:
            log_x_start = index_to_log_onehot(x, self.num_classes)

            t, pt = self.sample_time(b, device, 'importance')

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t, out_dict)

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            loss = kl / pt + kl_prior

            return -loss

    def forward(self, cond, f0=None, uv=None, nonpadding=None, ret=None, infer=False, dyn_clip=None):
        b = cond.shape[0]
        device = cond.device
        out_dict = {}
        if not infer:
            return self.mixed_loss(f0, uv, cond, nonpadding, out_dict)
        else:
            shape = (cond.shape[0], 1, cond.shape[2])
            x = self.sample(cond, shape, device, out_dict, dyn_clip)
            x = x.transpose(1, 2)
        return x
    
    def mixed_loss(self, f0, uv, cond, nonpadding, out_dict):
        # nonpadding [b, t]
        b = f0.shape[0]
        device = f0.device
        t, pt = self.sample_time(b, device, 'uniform')

        x_num = f0
        x_cat = uv
        
        x_num_t = x_num
        log_x_cat_t = x_cat
        # if x_num.shape[1] > 0:
        noise = torch.randn_like(x_num)
        x_num_t = self.gaussian_q_sample(x_num, t, noise=noise)
        # if x_cat.shape[1] > 0:
        log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes)
        log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t)
        x_cat_t = log_onehot_to_index(log_x_cat_t).long()
        # x_in = torch.cat([x_num_t, x_cat_t], dim=1)

        model_out = self._denoise_fn(
            x_num_t, x_cat_t,
            t,
            cond,
            nonpadding
        )

        model_out_num = model_out[:, :1, :] # f0
        model_out_cat = model_out[:, 1:, :] # uv
        
        # nll_loss = self.multinomial_kl(model_out_cat, log_x_cat)
        # nll_loss = sum_except_batch_with_padding(nll_loss, nonpadding_mask=nonpadding)
        loss_multi = torch.zeros((1,)).float()
        loss_gauss = torch.zeros((1,)).float()

        loss_multi = self._multinomial_loss(model_out_cat, log_x_cat, log_x_cat_t, t, pt, out_dict, nonpadding=nonpadding)
        
        loss_gauss = self.gaussian_p_losses(model_out_num, noise, nonpadding=nonpadding * (uv == 0).float())

        # loss_multi = torch.where(out_dict['y'] == 1, loss_multi, 2 * loss_multi)
        # loss_gauss = torch.where(out_dict['y'] == 1, loss_gauss, 2 * loss_gauss)

        return loss_multi.mean(), loss_gauss.mean(), None
    
    @torch.no_grad()
    def mixed_elbo(self, x0, out_dict):
        b = x0.size(0)
        device = x0.device

        x_num = x0[:, :self.num_numerical_features]
        x_cat = x0[:, self.num_numerical_features:]
        has_cat = x_cat.shape[1] > 0
        if has_cat:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes).to(device)

        gaussian_loss = []
        xstart_mse = []
        mse = []
        mu_mse = []
        out_mean = []
        true_mean = []
        multinomial_loss = []
        for t in range(self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()
            noise = torch.randn_like(x_num)

            x_num_t = self.gaussian_q_sample(x_start=x_num, t=t_array, noise=noise)
            if has_cat:
                log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t_array)
            else:
                log_x_cat_t = x_cat

            model_out = self._denoise_fn(
                torch.cat([x_num_t, log_x_cat_t], dim=1),
                t_array,
                **out_dict
            )
            
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]

            kl = torch.tensor([0.0])
            if has_cat:
                kl = self.compute_Lt(
                    model_out=model_out_cat,
                    log_x_start=log_x_cat,
                    log_x_t=log_x_cat_t,
                    t=t_array,
                    out_dict=out_dict
                )

            out = self._vb_terms_bpd(
                model_out_num,
                x_start=x_num,
                x_t=x_num_t,
                t=t_array,
                clip_denoised=False
            )

            multinomial_loss.append(kl)
            gaussian_loss.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_num) ** 2))
            # mu_mse.append(mean_flat(out["mean_mse"]))
            out_mean.append(mean_flat(out["out_mean"]))
            true_mean.append(mean_flat(out["true_mean"]))

            eps = self._predict_eps_from_xstart(x_num_t, t_array, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        gaussian_loss = torch.stack(gaussian_loss, dim=1)
        multinomial_loss = torch.stack(multinomial_loss, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)
        # mu_mse = torch.stack(mu_mse, dim=1)
        out_mean = torch.stack(out_mean, dim=1)
        true_mean = torch.stack(true_mean, dim=1)



        prior_gauss = self._prior_gaussian(x_num)

        prior_multin = torch.tensor([0.0])
        if has_cat:
            prior_multin = self.kl_prior(log_x_cat)

        total_gauss = gaussian_loss.sum(dim=1) + prior_gauss
        total_multin = multinomial_loss.sum(dim=1) + prior_multin
        return {
            "total_gaussian": total_gauss,
            "total_multinomial": total_multin,
            "losses_gaussian": gaussian_loss,
            "losses_multinimial": multinomial_loss,
            "xstart_mse": xstart_mse,
            "mse": mse,
            # "mu_mse": mu_mse
            "out_mean": out_mean,
            "true_mean": true_mean
        }

    @torch.no_grad()
    def gaussian_ddim_step(
        self,
        model_out_num,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        eta=0.0
    ):
        out = self.gaussian_p_mean_variance(
            model_out_num,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=None,
        )

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return sample
    
    @torch.no_grad()
    def gaussian_ddim_sample(
        self,
        noise,
        T,
        out_dict,
        eta=0.0
    ):
        x = noise
        b = x.shape[0]
        device = x.device
        for t in reversed(range(T)):
            print(f'Sample timestep {t:4d}', end='\r')
            t_array = (torch.ones(b, device=device) * t).long()
            out_num = self._denoise_fn(x, t_array, **out_dict)
            x = self.gaussian_ddim_step(
                out_num,
                x,
                t_array
            )
        print()
        return x


    @torch.no_grad()
    def gaussian_ddim_reverse_step(
        self,
        model_out_num,
        x,
        t,
        clip_denoised=False,
        eta=0.0
    ):
        assert eta == 0.0, "Eta must be zero."
        out = self.gaussian_p_mean_variance(
            model_out_num,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=None,
            model_kwargs=None,
        )

        eps = (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract(self.alphas_cumprod_next, t, x.shape)

        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return mean_pred

    @torch.no_grad()
    def gaussian_ddim_reverse_sample(
        self,
        x,
        T,
        out_dict,
    ):
        b = x.shape[0]
        device = x.device
        for t in range(T):
            print(f'Reverse timestep {t:4d}', end='\r')
            t_array = (torch.ones(b, device=device) * t).long()
            out_num = self._denoise_fn(x, t_array, **out_dict)
            x = self.gaussian_ddim_reverse_step(
                out_num,
                x,
                t_array,
                eta=0.0
            )
        print()

        return x


    @torch.no_grad()
    def multinomial_ddim_step(
        self,
        model_out_cat,
        log_x_t,
        t,
        out_dict,
        eta=0.0
    ):
        # not ddim, essentially
        log_x0 = self.predict_start(model_out_cat, log_x_t=log_x_t, t=t, out_dict=out_dict)

        alpha_bar = extract(self.alphas_cumprod, t, log_x_t.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, log_x_t.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        coef1 = sigma
        coef2 = alpha_bar_prev - sigma * alpha_bar
        coef3 = 1 - coef1 - coef2
        

        log_ps = torch.stack([
            torch.log(coef1) + log_x_t,
            torch.log(coef2) + log_x0,
            torch.log(coef3) - torch.log(self.num_classes_expanded)
        ], dim=2)

        log_prob = torch.logsumexp(log_ps, dim=2)

        out = self.log_sample_categorical(log_prob)

        return out

    @torch.no_grad()
    def sample_ddim(self, num_samples, y_dist):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
            log_z = self.log_sample_categorical(uniform_logits)

        y = torch.multinomial(
            y_dist,
            num_samples=b,
            replacement=True
        )
        out_dict = {'y': y.long().to(device)}
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(),
                t,
                **out_dict
            )
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            z_norm = self.gaussian_ddim_step(model_out_num, z_norm, t, clip_denoised=False)
            if has_cat:
                log_z = self.multinomial_ddim_step(model_out_cat, log_z, t, out_dict)

        print()
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict
    

    @torch.no_grad()
    def sample(self, cond, shape, device, out_dict, dyn_clip):
        nonpadding = torch.ones([shape[0], shape[-1]], device=device)
        uniform_logits = torch.zeros(shape, device=device)
        b = shape[0]
        log_z_uv = self.log_sample_categorical(uniform_logits)
        z_f0 = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sample time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                z_f0, log_onehot_to_index(log_z_uv).long(),
                t,
                cond,
                nonpadding
            )
            model_out_num = model_out[:, :1, :]
            model_out_cat = model_out[:, 1:, :]
            z_f0 = self.gaussian_p_sample(z_f0, torch.full((b,), i, device=device, dtype=torch.long), model_out_num, clip_denoised=True, dyn_clip=dyn_clip)
            log_z_uv = self.p_sample(model_out_cat, log_z_uv, t, out_dict)
        z_cat = log_onehot_to_index(log_z_uv).unsqueeze(dim=1)
        sample = torch.cat([z_f0, z_cat], dim=1)
        return sample
    
    def sample_all(self, num_samples, batch_size, y_dist, ddim=False):
        if ddim:
            print('Sample using DDIM.')
            sample_fn = self.sample_ddim
        else:
            sample_fn = self.sample
        
        b = batch_size

        all_y = []
        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            sample, out_dict = sample_fn(b, y_dist)
            mask_nan = torch.any(sample.isnan(), dim=1)
            sample = sample[~mask_nan]
            out_dict['y'] = out_dict['y'][~mask_nan]

            all_samples.append(sample)
            all_y.append(out_dict['y'].cpu())
            if sample.shape[0] != b:
                raise FoundNANsError
            num_generated += sample.shape[0]

        x_gen = torch.cat(all_samples, dim=0)[:num_samples]
        y_gen = torch.cat(all_y, dim=0)[:num_samples]

        return x_gen, y_gen