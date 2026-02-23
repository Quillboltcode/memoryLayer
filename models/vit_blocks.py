import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional
from .HopfieldLayer import HopfieldLayer


class MemoryAugmentedViTBlock(nn.Module):
    """
    Vision Transformer Block with Hopfield-based associative memory.
    
    Inserts a Hopfield layer either after self-attention or after FFN
    to augment the representation with long-term memory.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mem_size: int = 1000,
        hopfield_position: Literal["after_attn", "after_ffn"] = "after_attn",
        beta: float = 0.125,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.hopfield_position = hopfield_position
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, 
            num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.hopfield = HopfieldLayer(
            dim=dim,
            num_heads=num_heads,
            mem_size=mem_size,
            beta=beta,
            dropout=dropout
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, update_memory: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, D)
            update_memory: Whether to update memory with Hebbian learning
            
        Returns:
            Output tensor of shape (B, N, D)
        """
        if self.hopfield_position == "after_attn":
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            x = x + self.hopfield(self.norm2(x), update_memory=update_memory)
            x = x + self.ffn(self.norm3(x))
        else:
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            x = x + self.ffn(self.norm2(x))
            x = x + self.hopfield(self.norm3(x), update_memory=update_memory)
        
        return x
    
    def set_memory_mode(self, mode: Literal["associative", "class_prototype"]):
        """Switch memory mode."""
        self.hopfield.set_memory_mode(mode)
    
    def store_class_prototypes(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor, 
        num_classes: int
    ):
        """Store class prototypes in Hopfield layer."""
        self.hopfield.store_class_prototypes(features, labels, num_classes)
    
    def retrieve_class_prototype(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve closest class prototype."""
        return self.hopfield.retrieve_class_prototype(query)


class MemoryAugmentedViTBlockWithFFN(nn.Module):
    """
    Alternative implementation with separate FFN module for more flexibility.
    Used when integrating with timm models.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mem_size: int = 1000,
        hopfield_position: Literal["after_attn", "after_ffn"] = "after_attn",
        beta: float = 0.125,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.hopfield_position = hopfield_position
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.hopfield = HopfieldLayer(
            dim=dim,
            num_heads=num_heads,
            mem_size=mem_size,
            beta=beta,
            dropout=dropout
        )
        
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        attn_fn=None,
        update_memory: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            attn_fn: Optional custom attention function (for timm integration)
            update_memory: Whether to update memory with Hebbian learning
        """
        if attn_fn is not None:
            x = x + attn_fn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        else:
            x = x + self._default_attn(self.norm1(x))
        
        if self.hopfield_position == "after_attn":
            x = x + self.hopfield(self.norm2(x), update_memory=update_memory)
            x = x + self.mlp(self.norm3(x))
        else:
            x = x + self.mlp(self.norm2(x))
            x = x + self.hopfield(self.norm3(x), update_memory=update_memory)
        
        return x
    
    def _default_attn(self, x: torch.Tensor):
        """Fallback attention if no custom function provided."""
        return nn.MultiheadAttention(
            self.dim, self.num_heads, batch_first=True
        )(x, x, x)[0]
