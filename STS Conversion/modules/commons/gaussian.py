import math
import torch


def gaussian_loss(y_hat, y, log_std_min=-9.0):
    """

    :param y_hat: [B, T, 160]
    :param y: [B, T, 80]
    :param log_std_min:
    :return:
    """
    # assert y_hat.dim() == 3
    # assert y_hat.size(1) == 2
    # (B x T x C)
    # y_hat = y_hat.transpose(1, 2)
    B, T, H_mul_2 = y_hat.shape
    y_hat = y_hat.reshape(B, T, -1, 2)  # [B, T, 80, 2]
    mean = y_hat[:, :, :, 0]
    log_std = torch.clamp(y_hat[:, :, :, 1], min=log_std_min)
    log_probs = -0.5 * (
            - math.log(2.0 * math.pi) - 2. * log_std - (y - mean).pow(2) * (-2.0 * log_std).exp())
    return log_probs


def sample_from_gaussian(y_hat, temp=1.0):
    B, T, H_mul_2 = y_hat.shape
    y_hat = y_hat.reshape(B, T, -1, 2)  # [B, T, 80, 2]
    mean = y_hat[:, :, :, 0]
    log_std = y_hat[:, :, :, 1]
    sample = mean + torch.randn_like(log_std) * log_std.exp() * temp
    return sample


def KL_gaussians(stu_out, tea_out, log_std_min=-6.0, regularization=True):
    B, T = stu_out.shape[:2]
    stu_out = stu_out.reshape(B, T, -1, 2)  # [B, T, 80, 2]
    tea_out = tea_out.reshape(B, T, -1, 2)  # [B, T, 80, 2]
    mu_q, logs_q = stu_out[:, :, :, 0], stu_out[:, :, :, 1]
    mu_p, logs_p = tea_out[:, :, :, 0], tea_out[:, :, :, 1]

    # KL (q || p)
    # q ~ N(mu_q, logs_q.exp_()), p ~ N(mu_p, logs_p.exp_())
    logs_q_org = logs_q
    logs_p_org = logs_p
    logs_q = torch.clamp(logs_q, min=log_std_min)
    logs_p = torch.clamp(logs_p, min=log_std_min)
    KL_loss = (logs_p - logs_q) + 0.5 * (
            (torch.exp(2. * logs_q) + torch.pow(mu_p - mu_q, 2)) * torch.exp(-2. * logs_p) - 1.)
    if regularization:
        reg_loss = torch.pow(logs_q_org - logs_p_org, 2)
    else:
        reg_loss = None
    loss_tot = KL_loss + reg_loss * 4.
    return loss_tot, KL_loss, reg_loss
