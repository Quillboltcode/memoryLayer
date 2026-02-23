import torch
import torch.nn as nn
from typing import Optional, Literal


class LinearProbeViT(nn.Module):
    """
    Linear Probe on frozen ViT backbone.
    
    The backbone is frozen and only a linear head is trained.
    This is a strong baseline for limited data scenarios.
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1000,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        import timm
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.embed_dim = self.backbone.embed_dim
        
        self.head = nn.Linear(self.embed_dim, num_classes)
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        if isinstance(features, tuple):
            features = features[0]
        
        cls_token = features[:, 0]
        return self.head(cls_token)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        features = self.backbone.forward_features(x)
        if isinstance(features, tuple):
            features = features[0]
        return features[:, 0]
    
    def get_trainable_parameters(self):
        """Get only trainable parameters (the head)."""
        return filter(lambda p: p.requires_grad, self.parameters())


class LinearProbeWithMLP(nn.Module):
    """
    Linear probe with optional MLP head (non-linear probe).
    
    More expressive than linear probe while still being simple.
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1000,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        import timm
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.embed_dim = self.backbone.embed_dim
        
        if hidden_dim is None:
            hidden_dim = self.embed_dim
        
        if hidden_dim == self.embed_dim:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        
        self._freeze_backbone()
    
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        if isinstance(features, tuple):
            features = features[0]
        cls_token = features[:, 0]
        return self.head(cls_token)


def create_linear_probe(
    backbone_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    num_classes: int = 1000,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.1,
    use_mlp: bool = False,
) -> nn.Module:
    """Factory function to create linear probe model."""
    if use_mlp:
        return LinearProbeWithMLP(
            backbone_name=backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    return LinearProbeViT(
        backbone_name=backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
