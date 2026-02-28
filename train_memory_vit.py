#!/usr/bin/env python3
"""
Training script for Memory-Augmented ViT and baseline models.

Supports:
- MemoryAugmentedViT (Hopfield-augmented)
- LinearProbeViT (frozen backbone + linear head)
- VPT-Shallow, VPT-Deep (visual prompt tuning)
- FrozenBackboneViT (full fine-tuning)

Usage:
    # Train Memory-Augmented ViT
    python train_memory_vit.py --config-path config_memory_vit.yaml
    
    # Train with specific settings
    python train_memory_vit.py --model memory_vit --dataset cifar10 --epochs 100
    
    # Train baseline (Linear Probe)
    python train_memory_vit.py --model linear_probe --dataset cifar10 --epochs 50
    
    # Few-shot learning
    python train_memory_vit.py --model memory_vit --dataset cifar10 --few-shot --n-shot 5
"""

import os
import sys
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from omegaconf import OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_env_overrides(config: dict) -> dict:
    """Merge environment variable overrides into config."""
    env_mappings = {
        'MEMORY_VLM_EPOCHS': ('training', 'epochs'),
        'MEMORY_VLM_BATCH_SIZE': ('training', 'batch_size'),
        'MEMORY_VLM_LR': ('training', 'learning_rate'),
        'MEMORY_VLM_MODEL': ('model', 'name'),
        'MEMORY_VLM_DATASET': ('data', 'dataset'),
        'MEMORY_VLM_MEM_SIZE': ('model', 'mem_size'),
        'MEMORY_VLM_FREEZE': ('model', 'freeze_backbone'),
    }
    
    for env_var, config_path in env_mappings.items():
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Memory-Augmented ViT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--config-path',
        type=str,
        default='config_memory_vit.yaml',
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        choices=['memory_vit', 'linear_probe', 'vpt_shallow', 'vpt_deep', 'frozen_vit'],
        help='Model type (overrides config)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=['cifar10', 'cifar100', 'rafdb'],
        help='Dataset (overrides config)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--few-shot',
        action='store_true',
        help='Use few-shot learning mode'
    )
    parser.add_argument(
        '--n-shot',
        type=int,
        default=5,
        help='Number of shots for few-shot learning'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory'
    )
    
    return parser.parse_args()


def create_model(config: dict, device: torch.device):
    """Create model based on config."""
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    backbone = config['model'].get('backbone', 'vit_base_patch16_224')
    pretrained = config['model'].get('pretrained', True)
    
    if model_name == 'memory_vit':
        from models.memory_vit import MemoryAugmentedViT
        model = MemoryAugmentedViT(
            backbone_name=backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            mem_size=config['model'].get('mem_size', 1000),
            hopfield_position=config['model'].get('hopfield_position', 'after_attn'),
            beta=config['model'].get('beta', 0.125),
            dropout=config['model'].get('dropout', 0.1),
            freeze_backbone=config['model'].get('freeze_backbone', True),
        )
    elif model_name == 'linear_probe':
        from models.baselines.linear_probe import LinearProbeViT
        model = LinearProbeViT(
            backbone_name=backbone,
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif model_name == 'vpt_shallow':
        from models.baselines.vpt import VPTShallow
        model = VPTShallow(
            backbone_name=backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            num_prompts=config['model'].get('num_prompts', 10),
            freeze_backbone=config['model'].get('freeze_backbone', True),
        )
    elif model_name == 'vpt_deep':
        from models.baselines.vpt import VPTDeep
        model = VPTDeep(
            backbone_name=backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            num_prompts=config['model'].get('num_prompts', 10),
            freeze_backbone=config['model'].get('freeze_backbone', True),
        )
    elif model_name == 'frozen_vit':
        from models.memory_vit import FrozenBackboneViT
        model = FrozenBackboneViT(
            backbone_name=backbone,
            pretrained=pretrained,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    return model


def create_dataloaders(config: dict, few_shot: bool = False, n_shot: int = 5):
    """Create dataloaders based on config."""
    dataset_name = config['data']['dataset']
    root = config['data'].get('root', './data')
    image_size = config['data'].get('image_size', 224)
    batch_size = config['training'].get('batch_size', 64)
    num_workers = config['data'].get('num_workers', 4)
    
    if dataset_name == 'cifar10':
        if few_shot:
            from data.cifar import FewShotCIFAR10
            train_dataset = FewShotCIFAR10(
                root=root,
                n_way=10,
                n_shot=n_shot,
                image_size=image_size,
            )
            num_classes = 10
        else:
            from data.cifar import CIFAR10Dataset
            train_dataset = CIFAR10Dataset(
                root=root,
                train=True,
                image_size=image_size,
                augment=True,
            )
            num_classes = 10
        
        from data.cifar import CIFAR10Dataset
        test_dataset = CIFAR10Dataset(
            root=root,
            train=False,
            image_size=image_size,
            augment=False,
        )
    
    elif dataset_name == 'cifar100':
        if few_shot:
            from data.cifar import FewShotCIFAR100
            train_dataset = FewShotCIFAR100(
                root=root,
                n_way=100,
                n_shot=n_shot,
                image_size=image_size,
            )
            num_classes = 100
        else:
            from data.cifar import CIFAR100Dataset
            train_dataset = CIFAR100Dataset(
                root=root,
                train=True,
                image_size=image_size,
                augment=True,
            )
            num_classes = 100
        
        from data.cifar import CIFAR100Dataset
        test_dataset = CIFAR100Dataset(
            root=root,
            train=False,
            image_size=image_size,
            augment=False,
        )
    
    elif dataset_name == 'rafdb':
        from data.rafdb import RAFDBDataset
        train_dataset = RAFDBDataset(
            root=root,
            split='train',
            image_size=image_size,
            augment=True,
        )
        test_dataset = RAFDBDataset(
            root=root,
            split='test',
            image_size=image_size,
            augment=False,
        )
        num_classes = 7
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config['data'].get('pin_memory', True),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config['data'].get('pin_memory', True),
    )
    
    config['model']['num_classes'] = num_classes
    
    return train_loader, test_loader, num_classes


def setup_logging(config: dict):
    """Setup logging (WandB)."""
    log_config = config.get('logging', {})
    
    if log_config.get('wandb', False):
        try:
            import wandb
            wandb.init(
                project=log_config.get('project', 'memory-vlm'),
                name=log_config.get('name', 'memory_vit_exp'),
                config=config,
            )
            return wandb
        except Exception as e:
            print(f"Warning: WandB init failed: {e}")
    
    return None


def print_model_freeze_status(model: nn.Module):
    """Print which model components are frozen vs trainable."""
    frozen_params = []
    trainable_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)
    
    print("\n" + "="*60)
    print("MODEL FREEZE STATUS")
    print("="*60)
    
    # Group by component
    components = {}
    for name in trainable_params:
        if 'hopfield' in name:
            comp = 'hopfield'
        elif 'head' in name:
            comp = 'head'
        else:
            comp = 'other_trainable'
        components.setdefault(comp, []).append(name)
    
    for name in frozen_params:
        if 'blocks' in name:
            comp = 'backbone_blocks'
        elif 'patch_embed' in name:
            comp = 'patch_embed'
        elif 'pos_drop' in name or 'pos_embed' in name:
            comp = 'pos_embed'
        elif 'norm' in name:
            comp = 'norm'
        else:
            comp = 'other_frozen'
        components.setdefault(comp, []).append(name)
    
    # Print trainable components
    print("\n[TRAINABLE PARAMETERS]")
    for comp in ['hopfield', 'head']:
        if comp in components:
            params = components[comp]
            total_params = sum(p.numel() for n, p in model.named_parameters() if n in params)
            print(f"  {comp}: {len(params)} tensors, {total_params:,} params")
    
    # Print frozen components
    print("\n[FROZEN PARAMETERS]")
    for comp in ['backbone_blocks', 'patch_embed', 'pos_embed', 'norm']:
        if comp in components:
            params = components[comp]
            total_params = sum(p.numel() for n, p in model.named_parameters() if n in params)
            print(f"  {comp}: {len(params)} tensors, {total_params:,} params")
    
    # Summary
    frozen_count = sum(p.numel() for n, p in model.named_parameters() if not p.requires_grad)
    trainable_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    total_count = frozen_count + trainable_count
    
    print("\n[SUMMARY]")
    print(f"  Trainable: {trainable_count:,} ({100*trainable_count/total_count:.2f}%)")
    print(f"  Frozen: {frozen_count:,} ({100*frozen_count/total_count:.2f}%)")
    print(f"  Total: {total_count:,}")
    print("="*60 + "\n")


def train(config: dict, device: torch.device, logger=None):
    """Main training function."""
    print(f"Config: {config}")
    print(f"Device: {device}")
    
    model = create_model(config, device)
    print(f"Model: {config['model']['name']}")
    
    print_model_freeze_status(model)
    
    train_loader, test_loader, num_classes = create_dataloaders(config)
    print(f"Dataset: {config['data']['dataset']} ({len(train_loader.dataset)} train, {len(test_loader.dataset)} test)")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training'].get('learning_rate', 1e-3),
        weight_decay=config['training'].get('weight_decay', 0.01),
    )
    
    scheduler = None
    if config['training'].get('scheduler') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training'].get('epochs', 100)
        )
    
    use_amp = config['training'].get('use_amp', True) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    best_acc = 0.0
    best_model_state = None
    num_epochs = config['training'].get('epochs', 100)
    output_dir = config.get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Early stopping
    early_stopping_patience = config.get('early_stopping', {}).get('patience', 10)
    early_stopping_min_delta = config.get('early_stopping', {}).get('min_delta', 0.0)
    no_improvement_count = 0
    early_stop = False
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = nn.functional.cross_entropy(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        if scheduler is not None:
            scheduler.step()
        
        if epoch % config.get('eval', {}).get('eval_interval', 1) == 0:
            model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc='Evaluating'):
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
            
            test_acc = 100. * test_correct / test_total
            
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
            
            if logger:
                logger.log({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'epoch': epoch,
                })
            
            # Early stopping check
            if test_acc > best_acc + early_stopping_min_delta:
                best_acc = test_acc
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, os.path.join(output_dir, 'best_model.pt'))
                no_improvement_count = 0
                print(f"  -> New best model! Accuracy: {best_acc:.2f}%")
            else:
                no_improvement_count += 1
                print(f"  -> No improvement for {no_improvement_count} evaluation(s)")
                
                if early_stopping_patience > 0 and no_improvement_count >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs!")
                    early_stop = True
        else:
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            if logger:
                logger.log({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'epoch': epoch,
                })
        
        if epoch % config.get('save_interval', 10) == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f'epoch_{epoch}.pt'))
        
        if early_stop:
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with accuracy: {best_acc:.2f}%")
    
    torch.save(model.state_dict(), os.path.join(output_dir, 'last_model.pt'))
    
    print(f"Training complete! Best Test Accuracy: {best_acc:.2f}%")
    
    if logger:
        logger.finish()
    
    return best_acc


def main():
    """Main entry point."""
    args = parse_args()
    
    if os.path.exists(args.config_path):
        config = load_config(args.config_path)
        config = merge_env_overrides(config)
    else:
        print(f"Config file not found: {args.config_path}, using defaults")
        config = get_default_config()
    
    if args.model is not None:
        config['model']['name'] = args.model
    if args.dataset is not None:
        config['data']['dataset'] = args.dataset
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    
    device = torch.device(args.device)
    
    logger = setup_logging(config)
    
    train(config, device, logger)


def get_default_config() -> dict:
    """Get default configuration."""
    return {
        'model': {
            'name': 'memory_vit',
            'backbone': 'vit_base_patch16_224',
            'pretrained': True,
            'num_classes': 10,
            'mem_size': 1000,
            'hopfield_position': 'after_attn',
            'beta': 0.125,
            'dropout': 0.1,
            'freeze_backbone': True,
            'num_prompts': 10,
        },
        'training': {
            'epochs': 100,
            'batch_size': 64,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'scheduler': 'cosine',
            'use_amp': True,
        },
        'data': {
            'dataset': 'cifar10',
            'root': './data',
            'image_size': 224,
            'num_workers': 4,
            'pin_memory': True,
        },
        'logging': {
            'project': 'memory-vlm',
            'name': 'memory_vit_exp',
            'wandb': False,
        },
        'eval': {
            'eval_interval': 1,
        },
        'early_stopping': {
            'patience': 10,
            'min_delta': 0.0,
        },
        'output_dir': 'outputs',
    }


if __name__ == '__main__':
    main()
