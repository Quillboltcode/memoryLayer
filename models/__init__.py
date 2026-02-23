"""
MemoryCLIP: Vision-Language Model with in-weight memory augmentation.
Also includes Hopfield-based memory-augmented ViT models.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

from .HopfieldLayer import HopfieldLayer
from .vit_blocks import MemoryAugmentedViTBlock
from .memory_vit import (
    MemoryAugmentedViT,
    LinearProbeViT,
    FrozenBackboneViT,
    create_memory_vit,
    create_linear_probe,
)


class QuickGELU(nn.Module):
    """Quick GELU activation."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class MemoryToken(nn.Module):
    """Learnable memory tokens that augment vision features."""
    
    def __init__(
        self,
        num_tokens: int = 64,
        embed_dim: int = 512,
        mem_alpha: float = 0.5,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.mem_alpha = mem_alpha
        
        self.memory_tokens = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.memory_proj = nn.Linear(embed_dim, embed_dim)
        
        nn.init.normal_(self.memory_tokens, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        memory = self.memory_tokens.expand(batch_size, -1, -1)
        
        memory = self.memory_proj(memory)
        
        return memory
    
    def init_memory(self, x: torch.Tensor):
        """Initialize memory tokens from image features."""
        with torch.no_grad():
            img_features = x[:, 1:, :].mean(dim=1)
            self.memory_tokens.data = img_features[:, :self.num_tokens, :].clone()


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for memory fusion."""
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        memory: torch.Tensor,
        vision: torch.Tensor,
    ) -> torch.Tensor:
        memory = memory + self.attn(
            self.norm1(memory),
            self.norm1(vision),
            self.norm1(vision),
        )[0]
        memory = memory + self.mlp(self.norm2(memory))
        return memory


class VisionEncoder(nn.Module):
    """Vision encoder with memory augmentation."""
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        num_memory_tokens: int = 64,
        mem_alpha: float = 0.5,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.memory_tokens = MemoryToken(
            num_tokens=num_memory_tokens,
            embed_dim=embed_dim,
            mem_alpha=mem_alpha,
        )
        
        self.cross_attn = CrossAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=drop_rate,
        )
        
        self.use_timm = False
        
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.memory_tokens = MemoryToken(
            num_tokens=num_memory_tokens,
            embed_dim=embed_dim,
            mem_alpha=mem_alpha,
        )
        
        self.cross_attn = CrossAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=drop_rate,
        )
        
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=drop_rate,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        
        return x


class TextEncoder(nn.Module):
    """Text encoder for CLIP-style training."""
    
    def __init__(
        self,
        vocab_size: int = 49408,
        max_len: int = 77,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_len, embed_dim)
        )
        self.drop = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
    
    def forward(self, texts: List[str], tokenizer=None) -> torch.Tensor:
        if tokenizer is None:
            max_len = min(max(len(t) for t in texts) + 2, self.max_len)
            input_ids = torch.zeros(len(texts), max_len, dtype=torch.long)
            
            for i, text in enumerate(texts):
                tokens = text.lower().split()[:max_len - 2]
                for j, token in enumerate(tokens):
                    hash_val = hash(token) % self.vocab_size
                    input_ids[i, j + 1] = hash_val + 100
                input_ids[i, len(tokens) + 1] = 1
        else:
            encoded = tokenizer(
                texts,
                padding='max_length',
                max_length=self.max_len,
                truncation=True,
                return_tensors='pt',
            )
            input_ids = encoded['input_ids'].to(next(self.parameters()).device)
        
        B = input_ids.shape[0]
        
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding[:, :x.size(1), :]
        x = self.drop(x)
        
        x = self.transformer(x)
        x = self.norm(x)
        
        x = self.proj(x)
        
        return x


class MemoryCLIP(nn.Module):
    """
    Memory-augmented CLIP-style Vision-Language Model.
    
    Combines a vision encoder with learnable memory tokens and a text encoder
    for contrastive learning.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        image_size: int = 224,
        patch_size: int = 16,
        vision_depth: int = 12,
        text_depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        num_memory_tokens: int = 64,
        mem_alpha: float = 0.5,
        vocab_size: int = 49408,
        max_text_len: int = 77,
        temperature: float = 0.07,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.num_memory_tokens = num_memory_tokens
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        
        self.vision_encoder = VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=vision_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            num_memory_tokens=num_memory_tokens,
            mem_alpha=mem_alpha,
        )
        
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            max_len=max_text_len,
            embed_dim=embed_dim,
            depth=text_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=drop_rate,
        )
        
        self.vision_proj = nn.Linear(embed_dim, embed_dim)
        self.text_proj = nn.Linear(embed_dim, embed_dim)
        
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.vision_proj.weight, std=0.02)
        nn.init.zeros_(self.vision_proj.bias)
        nn.init.normal_(self.text_proj.weight, std=0.02)
        nn.init.zeros_(self.text_proj.bias)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        vision_features = self.vision_encoder(images)
        
        if len(vision_features.shape) == 2:
            cls_features = vision_features
            memory_out = torch.zeros_like(cls_features)
        else:
            cls_features = vision_features[:, 0]
            memory_features = vision_features[:, 1:1 + self.num_memory_tokens]
            memory_out = self.vision_proj(memory_features.mean(dim=1))
        
        cls_out = self.vision_proj(cls_features)
        
        image_features = cls_out + 0.1 * memory_out
        
        image_features = F.normalize(image_features, dim=-1)
        
        return image_features
    
    def encode_text(self, texts: List[str], tokenizer=None) -> torch.Tensor:
        text_features = self.text_encoder(texts, tokenizer)
        
        if len(text_features.shape) == 3:
            text_features = text_features.mean(dim=1)
        
        text_features = self.text_proj(text_features)
        
        text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def forward(
        self,
        images: torch.Tensor,
        texts: List[str],
        tokenizer=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts, tokenizer)
        
        return image_features, text_features
    
    def get_similarity(
        self,
        images: torch.Tensor,
        texts: List[str],
        tokenizer=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts, tokenizer)
        
        logits_per_image = image_features @ text_features.t() * self.temperature
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    
    def contrastive_loss(
        self,
        images: torch.Tensor,
        texts: List[str],
        tokenizer=None,
    ) -> torch.Tensor:
        logits_per_image, logits_per_text = self.get_similarity(
            images, texts, tokenizer
        )
        
        batch_size = images.shape[0]
        labels = torch.arange(batch_size, device=images.device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2


def MemoryCLIP_ViT_B_16(num_memory_tokens: int = 64, **kwargs):
    """Factory function for ViT-B/16 variant."""
    return MemoryCLIP(
        embed_dim=768,
        image_size=224,
        patch_size=16,
        vision_depth=12,
        text_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_memory_tokens=num_memory_tokens,
        **kwargs,
    )


def MemoryCLIP_ViT_L_14(num_memory_tokens: int = 64, **kwargs):
    """Factory function for ViT-L/14 variant."""
    return MemoryCLIP(
        embed_dim=1024,
        image_size=224,
        patch_size=14,
        vision_depth=24,
        text_depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_memory_tokens=num_memory_tokens,
        **kwargs,
    )


__all__ = [
    'MemoryCLIP',
    'VisionEncoder',
    'TextEncoder',
    'MemoryToken',
    'CrossAttentionLayer',
    'MemoryCLIP_ViT_B_16',
    'MemoryCLIP_ViT_L_14',
    'HopfieldLayer',
    'MemoryAugmentedViT',
    'MemoryAugmentedViTBlock',
    'LinearProbeViT',
    'FrozenBackboneViT',
    'create_memory_vit',
    'create_linear_probe',
]
