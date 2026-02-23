#!/usr/bin/env python3
"""
Memory-VLM Training Script

Supports two modes:
1. Hydra (default): python train.py training.epochs=100
2. Simple config: python train.py --config-path config_simple.yaml

For Kaggle notebooks, use --config-path to load a simple YAML file.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

try:
    from omegaconf import OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False


def load_simple_config(config_path: str) -> dict:
    """Load simple YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_env_overrides(config: dict) -> dict:
    """Merge environment variable overrides into config."""
    env_mappings = {
        'MEMORY_VLM_EPOCHS': ('training', 'epochs'),
        'MEMORY_VLM_BATCH_SIZE': ('training', 'batch_size'),
        'MEMORY_VLM_LR': ('training', 'learning_rate'),
        'MEMORY_VLM_WEIGHT_DECAY': ('training', 'weight_decay'),
        'MEMORY_VLM_EMBED_DIM': ('model', 'embed_dim'),
        'MEMORY_VLM_MEMORY_TOKENS': ('model', 'num_memory_tokens'),
        'MEMORY_VLM_IMAGE_SIZE': ('model', 'image_size'),
        'MEMORY_VLM_PROJECT': ('logging', 'project'),
        'MEMORY_VLM_NAME': ('logging', 'name'),
    }
    
    for env_var, config_path in env_mappings.items():
        if isinstance(config_path, tuple):
            value = os.environ.get(env_var)
            if value is not None:
                section, key = config_path
                if section not in config:
                    config[section] = {}
                try:
                    config[section][key] = int(value) if '.' not in value else float(value)
                except ValueError:
                    config[section][key] = value
    
    return config


def parse_args():
    """Parse command line arguments with support for both Hydra and simple config."""
    parser = argparse.ArgumentParser(
        description='Train Memory-VLM model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hydra mode (default)
  python train.py training.epochs=100 training.batch_size=128
  
  # Simple config mode
  python train.py --config-path config_simple.yaml
  
  # With environment variables
  MEMORY_VLM_EPOCHS=50 python train.py --config-path config_simple.yaml
  
  # CPU training (for testing)
  python train.py --config-path config_simple.yaml --device cpu
        """
    )
    
    parser.add_argument(
        '--config-path',
        type=str,
        default=os.environ.get('CONFIG_FILE', None),
        help='Path to simple YAML config file (for Kaggle/notebooks)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override learning rate'
    )
    
    if HYDRA_AVAILABLE:
        parser.add_argument(
            'overrides',
            nargs='*',
            default=[],
            help='Hydra config overrides (e.g., training.epochs=100)'
        )
    
    return parser.parse_args()


def load_hydra_config(overrides: list) -> Optional[dict]:
    """Load config using Hydra."""
    if not HYDRA_AVAILABLE:
        return None
    
    try:
        import hydra
        from hydra import compose, initialize_config_dir
        from omegaconf import DictConfig, OmegaConf
        
        config_dir = str(Path(__file__).parent / 'configs')
        
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='default', overrides=overrides)
        
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        return cfg_dict if isinstance(cfg_dict, dict) else None
    except Exception as e:
        print(f"Warning: Failed to load Hydra config: {e}")
        return None


def get_config(args) -> dict:
    """Get configuration from either Hydra or simple config."""
    if args.config_path:
        config = load_simple_config(args.config_path)
        config = merge_env_overrides(config)
    else:
        if HYDRA_AVAILABLE and args.overrides:
            config = load_hydra_config(args.overrides)
        else:
            config_path = os.environ.get('CONFIG_FILE', 'config_simple.yaml')
            if os.path.exists(config_path):
                config = load_simple_config(config_path)
                config = merge_env_overrides(config)
            else:
                config = get_default_config()
    
    if config is None:
        config = get_default_config()
    
    training_config = config.get('training', {}) if isinstance(config.get('training'), dict) else {}
    
    if args.epochs is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['epochs'] = args.epochs
    
    if args.batch_size is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['batch_size'] = args.batch_size
    
    if args.lr is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['learning_rate'] = args.lr
    
    return config


def get_default_config() -> dict:
    """Get default configuration."""
    return {
        'model': {
            'name': 'memory_clip_vit_b_16',
            'embed_dim': 512,
            'num_memory_tokens': 64,
            'image_size': 224,
        },
        'training': {
            'epochs': 100,
            'batch_size': 256,
            'learning_rate': 5e-4,
            'weight_decay': 0.01,
        },
        'data': {
            'train_path': 'data/train',
            'val_path': 'data/val',
            'image_size': 224,
        },
        'logging': {
            'project': 'memory-vlm',
            'name': 'memory_clip',
            'wandb': True,
            'tensorboard': True,
        },
    }


def create_dummy_dataset(path: str, image_size: int = 224, num_samples: int = 100):
    """Create dummy dataset for testing."""
    from PIL import Image as PILImage
    import numpy as np
    
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_samples):
        img = PILImage.fromarray(
            np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        )
        img.save(str(path_obj / f'image_{i:05d}.jpg'))


class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing."""
    
    def __init__(self, path: str, image_size: int = 224, num_samples: int = 100):
        self.path = Path(path)
        self.image_size = image_size
        self.num_samples = num_samples
        
        if not self.path.exists() or len(list(self.path.glob('*.jpg'))) < num_samples:
            create_dummy_dataset(str(self.path), image_size, num_samples)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        from PIL import Image as PILImage
        import numpy as np
        
        img_path = self.path / f'image_{idx:05d}.jpg'
        image = PILImage.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image_np = np.array(image).transpose(2, 0, 1)
        image_tensor = torch.from_numpy(image_np).float() / 255.0 * 2.0 - 1.0
        
        text = f"sample text {idx}"
        return {'image': image_tensor, 'text': text}


def setup_logging(config: dict):
    """Setup logging (WandB, TensorBoard)."""
    log_config = config.get('logging', {})
    
    if log_config.get('wandb', False):
        try:
            import wandb
            wandb.init(
                project=log_config.get('project', 'memory-vlm'),
                name=log_config.get('name', 'memory_clip'),
            )
            return wandb
        except Exception as e:
            print(f"Warning: WandB init failed: {e}")
    
    return None


def train_epoch(model, dataloader, optimizer, device, epoch: int, config: dict, logger=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        texts = batch['text']
        
        optimizer.zero_grad()
        
        with autocast(device.type == 'cuda'):
            image_embeds, text_embeds = model(images, texts)
            
            logits_per_image = image_embeds @ text_embeds.t()
            logits_per_text = logits_per_image.t()
            
            labels = torch.arange(len(images), device=device)
            loss_img = nn.functional.cross_entropy(logits_per_image, labels)
            loss_text = nn.functional.cross_entropy(logits_per_text, labels)
            loss = (loss_img + loss_text) / 2
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if logger and batch_idx % 10 == 0:
            logger.log({'train_loss': loss.item(), 'epoch': epoch})
    
    return total_loss / num_batches


def main():
    """Main training function."""
    args = parse_args()
    config = get_config(args)
    
    print(f"Config: {config}")
    print(f"Device: {args.device}")
    
    device = torch.device(args.device)
    
    logger = setup_logging(config)
    
    try:
        from models import MemoryCLIP
        model = MemoryCLIP(
            embed_dim=config['model'].get('embed_dim', 512),
            image_size=config['model'].get('image_size', 224),
            num_memory_tokens=config['model'].get('num_memory_tokens', 64),
        ).to(device)
    except ImportError:
        print("Creating simple model for testing...")
        model = nn.Sequential(
            nn.Linear(3 * 224 * 224, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        ).to(device)
    
    train_data = DummyDataset(
        config.get('data', {}).get('train_path', 'data/train'),
        config.get('data', {}).get('image_size', 224),
    )
    train_loader = DataLoader(
        train_data,
        batch_size=config.get('training', {}).get('batch_size', 256),
        shuffle=True,
        num_workers=0,
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('training', {}).get('learning_rate', 5e-4),
        weight_decay=config.get('training', {}).get('weight_decay', 0.01),
    )
    
    epochs = config.get('training', {}).get('epochs', 100)
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, epoch, config, logger)
        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")
        
        if logger:
            logger.log({'epoch_loss': loss, 'epoch': epoch})
    
    print("Training complete!")
    
    if logger:
        logger.finish()


if __name__ == '__main__':
    main()
