import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register

@register('afno')
class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        #self.num_blocks = num_blocks
        #self.block_size = self.hidden_size // self.num_blocks
        #self.hard_thresholding_fraction = hard_thresholding_fraction
        #self.scale = 0.02

        self.rfc0 = nn.Conv2d(hidden_size, hidden_size, 1, groups=num_blocks)
        self.rfc1 = nn.Conv2d(hidden_size, hidden_size, 1, groups=num_blocks)

        self.ifc0 = nn.Conv2d(hidden_size, hidden_size, 1, groups=num_blocks)
        self.ifc1 = nn.Conv2d(hidden_size, hidden_size, 1, groups=num_blocks)
        #self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        #self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        #self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        #self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape


        #x = x.reshape(B, H, W, C)
        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        #x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])

        x1r = F.relu(self.rfc0(x.real) - self.ifc0(x.imag))
        x1i = F.relu(self.rfc0(x.imag) + self.ifc0(x.real))

        x2r = (self.rfc1(x1r) - self.ifc1(x1i))
        x2i = (self.rfc1(x1i) + self.ifc1(x1r))

        #o1_real = torch.zeros(x.shape, device=x.device)
        #o1_imag = torch.zeros(x.shape, device=x.device)
        #o2_real = torch.zeros(x.shape, device=x.device)
        #o2_imag = torch.zeros(x.shape, device=x.device)

        #o1_real = F.relu(
        #    torch.einsum('.bihw,bio->.bohw', x.real, self.w1[0]) - \
        #    torch.einsum('.bihw,bio->.bohw', x.imag, self.w1[1]) + \
        #    self.b1[0]
        #)

        #o1_imag = F.relu(
        #    torch.einsum('.bihw,bio->.bohw', x.imag, self.w1[0]) + \
        #    torch.einsum('.bihw,bio->.bohw', x.real, self.w1[1]) + \
        #    self.b1[1]
        #)

        #o2_real = (
        #    torch.einsum('.bihw,bio->.bohw', o1_real, self.w2[0]) - \
        #    torch.einsum('.bihw,bio->.bohw', o1_imag, self.w2[1]) + \
        #    self.b2[0]
        #)

        #o2_imag = (
        #    torch.einsum('.bihw,bio->.bohw', o1_imag, self.w2[0]) + \
        #    torch.einsum('.bihw,bio->.bohw', o1_real, self.w2[1]) + \
        #    self.b2[1]
        #)

        x = torch.stack([x2r, x2i], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        #x = x.reshape(B, C, x.shape[2], x.shape[3])
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        #x = x.reshape(B, N, C)
        x = x.type(dtype)
        return x + bias