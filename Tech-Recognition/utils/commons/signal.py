import numpy as np
import torch
import torch.nn as nn

def get_filter_2d(kernel, kernel_size, channels, no_grad=True):
    # Reshape to 2d depthwise convolutional weight
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)

    filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                       bias=False, padding=kernel_size // 2)

    filter.weight.data = kernel
    if no_grad:
        filter.weight.requires_grad = False

    return filter

def get_filter_1d(kernel, kernel_size, channels, no_grad=True):
    kernel = kernel.view(1, 1, kernel_size)
    kernel = kernel.repeat(channels, 1, 1)

    filter = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                       bias=False, padding=kernel_size // 2)

    filter.weight.data = kernel
    if no_grad:
        filter.weight.requires_grad = False

    return filter

def get_gaussian_kernel_2d(kernel_size, sigma):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * np.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel

def get_gaussian_kernel_1d(kernel_size, sigma):
    x_grid = torch.arange(kernel_size)
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / ((2. * np.pi) ** 0.5 * sigma)) * torch.exp(-(x_grid - mean) ** 2. / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel

def get_hann_kernel_1d(kernel_size, periodic=False):
    # periodic=False gives symmetric kernel, otherwise equivalent to hann(kernel_size + 1)
    return torch.hann_window(kernel_size, periodic)

def get_triangle_kernel_1d(kernel_size):
    kernel = torch.zeros(kernel_size)
    for idx in range(kernel_size):
        kernel[idx] = 1 - abs((idx - (kernel_size - 1) / 2) / ((kernel_size - 1) / 2))
    return kernel

def add_gaussian_noise(tensor, mean=0, std=1):
    noise = torch.randn(tensor.size()) * std + mean
    noisy_tensor = tensor + noise
    return noisy_tensor
