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

    def __init__(self, dim, mem_size=1000, beta=0.125, dropout=0.1, num_heads=None):
        super().__init__()
        self.dim = dim
        self.mem_size = mem_size
        self.num_heads = num_heads  # Kept for compatibility but not used

        # Learnable memory (critical fix)
        self.memory_keys = nn.Parameter(torch.randn(1, mem_size, dim))
        self.memory_vals = nn.Parameter(torch.randn(1, mem_size, dim))

        # Query projection
        self.to_q = nn.Linear(dim, dim, bias=False)

        # Temperature (learnable scale)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(beta)))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, update_memory=False):
        B, N, C = x.shape

        # Project queries
        q = self.to_q(x)  # (B, N, C)

        # Expand memory for batch
        mem_k = self.memory_keys.expand(B, -1, -1)  # (B, mem_size, C)
        mem_v = self.memory_vals.expand(B, -1, -1)  # (B, mem_size, C)

        # Compute attention: (B, N, mem_size)
        attn = torch.matmul(q, mem_k.transpose(-2, -1)) * self.log_temp.exp()
        attn = F.softmax(attn, dim=-1)

        # Retrieve and apply dropout
        out = torch.matmul(attn, mem_v)  # (B, N, C)
        out = self.dropout(out)

        # Residual + LayerNorm (pre-norm style)
        out = self.norm(x + out)
        return out

    def set_memory_mode(self, mode: str):
        """Set the memory mode: 'store', 'query', or 'kernel'."""
        self.memory_mode = mode

    def store_class_prototypes(self, features, labels, num_classes):
        """Store class prototypes in memory keys/values."""
        B, N, C = features.shape
        cls_token = features[:, 0]
        
        class_features = {}
        for i in range(B):
            label = labels[i].item() if labels.dim() == 1 else labels[i]
            if label not in class_features:
                class_features[label] = []
            class_features[label].append(cls_token[i])
        
        prototypes = torch.zeros(num_classes, C, device=features.device)
        for cls, feats in class_features.items():
            prototypes[cls] = torch.stack(feats).mean(dim=0)
        
        if prototypes.shape[0] <= self.mem_size:
            self.memory_vals.data[0, :num_classes] = prototypes

    def retrieve_class_prototype(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve closest class prototype from memory."""
        B, N, C = query.shape
        q = self.to_q(query)  # (B, N, C)
        cls_query = q[:, 0]  # Use CLS token
        
        # Get memory
        mem_k = self.memory_keys[0]  # (mem_size, C)
        mem_v = self.memory_vals[0]  # (mem_size, C)
        
        # Compute similarity
        attn = torch.matmul(cls_query, mem_k.transpose(-2, -1)) * self.log_temp.exp()
        attn = F.softmax(attn, dim=-1)
        
        # Retrieve prototype
        prototype = torch.matmul(attn, mem_v)  # (B, C)
        return prototype
