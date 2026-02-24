import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HopfieldLayer(nn.Module):
    """
    Modern Hopfield Network (Dense Associative Memory)
    Based on Ramsauer et al. 2021 (https://arxiv.org/abs/2008.06713)

    Key changes from original:
    - Memory is learnable (nn.Parameter), not buffers
    - Removed unused multi-head logic
    - Fixed dropout placement
    - Removed manual memory updates (handled by optimizer)
    - Configurable memory size
    """

    def __init__(self, dim, mem_size=1000, beta=0.125, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.mem_size = mem_size

        # Learnable memory (critical fix)
        self.memory_keys = nn.Parameter(torch.randn(1, mem_size, dim))
        self.memory_vals = nn.Parameter(torch.randn(1, mem_size, dim))

        # Query projection
        self.to_q = nn.Linear(dim, dim, bias=False)

        # Temperature (learnable scale)
        self.temp = nn.Parameter(torch.ones(1) * beta)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape

        # Project queries
        q = self.to_q(x)  # (B, N, C)

        # Expand memory for batch
        mem_k = self.memory_keys.expand(B, -1, -1)  # (B, mem_size, C)
        mem_v = self.memory_vals.expand(B, -1, -1)  # (B, mem_size, C)

        # Compute attention: (B, N, mem_size)
        attn = torch.matmul(q, mem_k.transpose(-2, -1)) * self.temp
        attn = F.softmax(attn, dim=-1)

        # Retrieve and apply dropout
        out = torch.matmul(attn, mem_v)  # (B, N, C)
        out = self.dropout(out)

        # Residual + LayerNorm (pre-norm style)
        out = self.norm(x + out)
        return out
