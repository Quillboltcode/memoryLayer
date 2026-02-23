import torch
import torch.nn as nn
from typing import Optional, List


class VPTShallow(nn.Module):
    """
    Visual Prompt Tuning - Shallow (VPT-Shallow).
    
    Adds learnable prompts only at the first layer (input space).
    Reference: Jia et al. 2022 - "Visual Prompt Tuning"
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1000,
        num_prompts: int = 10,
        prompt_dim: Optional[int] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        import timm
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.num_prompts = num_prompts
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.embed_dim = self.backbone.embed_dim
        self.head = nn.Linear(self.embed_dim, num_classes)
        
        if prompt_dim is None:
            prompt_dim = self.embed_dim
        self.prompt_dim = prompt_dim
        
        if prompt_dim != self.embed_dim:
            self.prompt_proj = nn.Linear(prompt_dim, self.embed_dim)
        else:
            self.prompt_proj = nn.Identity()
        
        self.prompts = nn.Parameter(
            torch.randn(1, num_prompts, prompt_dim) * 0.02
        )
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        if isinstance(features, tuple):
            features = features[0]
        
        cls_token = features[:, 0:1]
        patch_tokens = features[:, 1:]
        
        B = x.shape[0]
        prompts = self.prompts.expand(B, -1, -1)
        prompts = self.prompt_proj(prompts)
        
        features_with_prompts = torch.cat([cls_token, prompts, patch_tokens], dim=1)
        
        cls_output = features_with_prompts[:, 0]
        return self.head(cls_output)


class VPTDeep(nn.Module):
    """
    Visual Prompt Tuning - Deep (VPT-Deep).
    
    Adds learnable prompts at every transformer block.
    Reference: Jia et al. 2022 - "Visual Prompt Tuning"
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1000,
        num_prompts: int = 10,
        prompt_dim: Optional[int] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        import timm
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.num_prompts = num_prompts
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.embed_dim = self.backbone.embed_dim
        self.num_blocks = len(self.backbone.blocks)
        self.head = nn.Linear(self.embed_dim, num_classes)
        
        if prompt_dim is None:
            prompt_dim = self.embed_dim
        self.prompt_dim = prompt_dim
        
        if prompt_dim != self.embed_dim:
            self.prompt_proj = nn.Linear(prompt_dim, self.embed_dim)
        else:
            self.prompt_proj = nn.Identity()
        
        self.prompts = nn.Parameter(
            torch.randn(1, num_prompts, prompt_dim) * 0.02
        )
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def _insert_prompts(self, x: torch.Tensor) -> torch.Tensor:
        """Insert prompts at the beginning of the sequence."""
        B = x.shape[0]
        prompts = self.prompts.expand(B, -1, -1)
        prompts = self.prompt_proj(prompts)
        return torch.cat([prompts, x], dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        if hasattr(self.backbone, 'pos_drop'):
            x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        elif hasattr(self.backbone, 'patch_embed.proj'):
            x = x + self.backbone.pos_embed
        
        x = self._insert_prompts(x)
        
        for block in self.backbone.blocks:
            x = block(x)
        
        if hasattr(self.backbone, 'norm'):
            x = self.backbone.norm(x)
        
        cls_output = x[:, 0]
        return self.head(cls_output)


class VPTToken(nn.Module):
    """
    Visual Prompt Tuning with learnable prompt tokens.
    Alternative approach with dedicated prompt tokens.
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1000,
        num_prompts: int = 10,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        import timm
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.num_prompts = num_prompts
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.embed_dim = self.backbone.embed_dim
        self.head = nn.Linear(self.embed_dim, num_classes)
        
        self.prompt_tokens = nn.Parameter(
            torch.randn(1, num_prompts, self.embed_dim) * 0.02
        )
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        features = self.backbone.forward_features(x)
        if isinstance(features, tuple):
            features = features[0]
        
        prompts = self.prompt_tokens.expand(B, -1, -1)
        
        features = torch.cat([features[:, :1], prompts, features[:, 1:]], dim=1)
        
        cls_output = features[:, 0]
        return self.head(cls_output)


def create_vpt_model(
    variant: str = "shallow",
    backbone_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    num_classes: int = 1000,
    num_prompts: int = 10,
    prompt_dim: Optional[int] = None,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Factory function to create VPT model."""
    if variant == "shallow":
        return VPTShallow(
            backbone_name=backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            num_prompts=num_prompts,
            prompt_dim=prompt_dim,
            freeze_backbone=freeze_backbone,
        )
    elif variant == "deep":
        return VPTDeep(
            backbone_name=backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            num_prompts=num_prompts,
            prompt_dim=prompt_dim,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(f"Unknown VPT variant: {variant}")
