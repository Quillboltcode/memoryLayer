import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class HopfieldLayer(nn.Module):
    """
    Modern Hopfield Network (Dense Associative Memory)
    Based on Demircigil et al. 2021 / Ramsauer et al. 2021

    Key insight: Softmax attention with pattern retrieval via energy minimization
    
    Supports two memory modes:
    - "associative": Random memory with Hebbian updates (default)
    - "class_prototype": Store class centroids for classification
    """

    def __init__(
        self, 
        dim, 
        num_heads=8, 
        mem_size: int = 1000, 
        beta: float = 0.125, 
        dropout: float = 0.1,
        memory_mode: Literal["associative", "class_prototype"] = "associative"
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.beta = beta
        self.mem_size = mem_size
        self.memory_mode = memory_mode
        
        self.register_buffer("memory_keys", torch.randn(1, self.mem_size, dim))
        self.register_buffer("memory_vals", torch.randn(1, self.mem_size, dim))
        
        self.to_q = nn.Linear(dim, dim)
        self.temp = nn.Parameter(torch.ones(1) * beta)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
        self._memory_usage = torch.zeros(1, mem_size)
        self._prototype_labels: Optional[torch.Tensor] = None

    def forward(self, x, update_memory=False):
        B, N, C = x.shape
        
        q = self.to_q(x)
        
        mem_k = self.memory_keys.expand(B, -1, -1)
        mem_v = self.memory_vals.expand(B, -1, -1)
        
        attn = torch.matmul(q, mem_k.transpose(-2, -1)) * self.temp
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, mem_v)
        
        out = self.norm(x + out)
        
        if update_memory and self.training:
            self.update_memory_hebbian(x)
        
        return out
    
    def set_memory_mode(self, mode: Literal["associative", "class_prototype"]):
        """Switch between associative memory and class prototype modes."""
        self.memory_mode = mode
    
    def store_class_prototypes(self, features: torch.Tensor, labels: torch.Tensor, num_classes: int):
        """
        Compute and store class prototypes (mean features per class).
        
        Args:
            features: Feature tensor of shape (N, D)
            labels: Label tensor of shape (N,)
            num_classes: Total number of classes
        """
        if features.size(0) != labels.size(0):
            raise ValueError("Features and labels must have same batch size")
        
        self.memory_mode = "class_prototype"
        
        prototypes = torch.zeros(num_classes, self.dim, device=features.device)
        class_counts = torch.zeros(num_classes, device=features.device)
        
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                prototypes[c] = features[mask].mean(dim=0)
                class_counts[c] = mask.sum()
        
        mem_size = min(num_classes, self.mem_size)
        self.memory_keys[:, :mem_size] = prototypes[:mem_size].unsqueeze(0)
        self.memory_vals[:, :mem_size] = prototypes[:mem_size].unsqueeze(0)
        
        self._prototype_labels = torch.arange(num_classes, device=features.device)
    
    def retrieve_class_prototype(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve closest class prototype for query features.
        
        Args:
            query: Query tensor of shape (B, D) or (B, N, D)
            
        Returns:
            Class predictions of shape (B,) or (B, N,)
        """
        if query.dim() == 3:
            query = query.mean(dim=1)
        
        q = self.to_q(query)
        
        mem_k = self.memory_keys.expand(q.size(0), -1, -1)
        attn = torch.matmul(q, mem_k.transpose(-2, -1)) * self.temp
        attn = F.softmax(attn, dim=-1)
        
        return attn.argmax(dim=-1)

    def update_memory_hebbian(self, x, usage_threshold=0.1):
        """
        Hebbian update: patterns that co-occur strengthen connections
        More biologically plausible, no gradients needed
        """
        with torch.no_grad():
            # Compute pattern usage (which memory slots are accessed)
            q = self.to_q(x.mean(dim=1, keepdim=True))  # Aggregate batch
            attn = torch.matmul(q, self.memory_keys.transpose(-2, -1))
            usage = attn.softmax(dim=-1).mean(dim=1)  # (1, mem_size)

            # Find least used slots to replace
            _, replace_idx = torch.topk(usage, k=x.size(0), largest=False)

            # Store new patterns (simple moving average)
            new_patterns = x.mean(dim=1)  # (B, C)
            new_patterns = x.mean(dim=1)
            replace_idx = replace_idx.squeeze(0)
            self.memory_keys[0, replace_idx] = (
                0.9 * self.memory_keys[0, replace_idx] + 0.1 * new_patterns
            )
            self.memory_vals[0, replace_idx] = self.memory_keys[0, replace_idx].clone()
    
    def reset_memory(self):
        """Reset memory to random initialization."""
        with torch.no_grad():
            self.memory_keys = torch.randn(1, self.mem_size, self.dim)
            self.memory_vals = torch.randn(1, self.mem_size, self.dim)
            self._memory_usage = torch.zeros(1, self.mem_size)
    
    def get_memory_usage(self) -> torch.Tensor:
        """Return current memory usage statistics."""
        return self._memory_usage.clone()
