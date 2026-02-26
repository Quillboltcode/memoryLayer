import torch
import torch.nn as nn
from typing import Optional, Literal
import timm


class HopfieldBlockWrapper(nn.Module):
    """Wraps a standard ViT block with Hopfield augmentation"""
    def __init__(self, vit_block, hopfield_layer, position="after_attn"):
        super().__init__()
        self.vit_block = vit_block
        self.hopfield_layer = hopfield_layer
        self.position = position
        
        self.vit_block.drop_path = nn.Identity()

    def forward(self, x):
        x_residual = x
        
        x = self.vit_block.norm1(x)
        x = self.vit_block.attn(x)
        
        if self.position == "after_attn":
            x = x_residual + x
            x = x + self.hopfield_layer(x)
        elif self.position == "residual":
            x = x_residual + x + self.hopfield_layer(x_residual)
        elif self.position == "after_mlp":
            x = x_residual + x
            
        x_residual = x
        x = self.vit_block.norm2(x)
        x = self.vit_block.mlp(x)
        
        if self.position == "after_mlp":
            x = x_residual + x + self.hopfield_layer(x_residual + x)
        else:
            x = x_residual + x
            
        return x


class MemoryAugmentedViT(nn.Module):
    """
    Vision Transformer with Hopfield-based associative memory.
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1000,
        mem_size: int = 1000,
        hopfield_position: str = "after_attn",
        beta: float = 0.125,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.mem_size = mem_size
        self.hopfield_position = hopfield_position
        self.freeze_backbone = freeze_backbone
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.embed_dim = self.backbone.embed_dim
        
        self._insert_hopfield_layers(mem_size, hopfield_position, beta, dropout)
        
        self.head = nn.Linear(self.embed_dim, num_classes)
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _insert_hopfield_layers(self, mem_size, position, beta, dropout):
        """Insert Hopfield layers into the ViT blocks."""
        from .HopfieldLayer import HopfieldLayer
        
        num_blocks = len(self.backbone.blocks)
        new_blocks = []
        
        for i in range(num_blocks):
            hopfield = HopfieldLayer(
                dim=self.embed_dim,
                mem_size=mem_size,
                beta=beta,
                dropout=dropout
            )
            wrapped_block = HopfieldBlockWrapper(
                self.backbone.blocks[i],
                hopfield,
                position=position
            )
            new_blocks.append(wrapped_block)
        
        self.backbone.blocks = nn.Sequential(*new_blocks)
    
    def _freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if 'hopfield' not in name and 'head' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def set_memory_mode(self, mode: str):
        for block in self.backbone.blocks:
            block.hopfield_layer.set_memory_mode(mode)
    
    def store_class_prototypes(self, features, labels, num_classes):
        for block in self.backbone.blocks:
            block.hopfield_layer.store_class_prototypes(features, labels, num_classes)
    
    def forward(self, x, return_features: bool = False):
        features = self.backbone.forward_features(x)
        
        if isinstance(features, tuple):
            features = features[0]
        
        cls_token = features[:, 0]
        
        logits = self.head(cls_token)
        
        if return_features:
            return features, logits
        return logits
    
    def extract_features(self, x):
        features = self.backbone.forward_features(x)
        if isinstance(features, tuple):
            features = features[0]
        return features
    
    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class LinearProbeViT(nn.Module):
    """Linear probe on frozen ViT."""
    
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1000,
    ):
        super().__init__()
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.embed_dim = self.backbone.embed_dim
        
        self.head = nn.Linear(self.embed_dim, num_classes)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        if isinstance(features, tuple):
            features = features[0]
        cls_token = features[:, 0]
        return self.head(cls_token)
    
    def extract_features(self, x):
        features = self.backbone.forward_features(x)
        if isinstance(features, tuple):
            features = features[0]
        return features[:, 0]


class FrozenBackboneViT(nn.Module):
    """ViT with frozen backbone and trainable head."""
    
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1000,
    ):
        super().__init__()
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        for param in self.backbone.head.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)


def create_memory_vit(
    backbone_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    num_classes: int = 1000,
    mem_size: int = 1000,
    hopfield_position: str = "after_attn",
    freeze_backbone: bool = True,
    **kwargs
):
    return MemoryAugmentedViT(
        backbone_name=backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
        mem_size=mem_size,
        hopfield_position=hopfield_position,
        freeze_backbone=freeze_backbone,
        **kwargs
    )


def create_linear_probe(
    backbone_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    num_classes: int = 1000,
):
    return LinearProbeViT(
        backbone_name=backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
