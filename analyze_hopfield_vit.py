"""
Hopfield-Augmented ViT Analysis Script

Comprehensive analysis tools for understanding Hopfield-augmented Vision Transformers.
Based on docs/analysis.md framework.

Usage:
    # With config file (recommended) - auto-detects config from checkpoint dir
    python analyze_hopfield_vit.py --checkpoint outputs/best_model.pt
    
    # Explicit config file
    python analyze_hopfield_vit.py --checkpoint outputs/best_model.pt --config config.yaml
    
    # Override settings
    python analyze_hopfield_vit.py --checkpoint outputs/best_model.pt --dataset cifar100 --batch_size 64
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

from models.memory_vit import MemoryAugmentedViT


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model(config: dict):
    """Create model based on config dict (from YAML or manual)."""
    model_cfg = config.get('model', config)  # Support both flat and nested config
    
    model = MemoryAugmentedViT(
        backbone_name=model_cfg.get('backbone', 'vit_base_patch16_224'),
        pretrained=False,
        num_classes=model_cfg.get('num_classes', 10),
        mem_size=model_cfg.get('mem_size', 1000),
        hopfield_position=model_cfg.get('hopfield_position', 'after_attn'),
        beta=model_cfg.get('beta', 0.125),
        dropout=model_cfg.get('dropout', 0.1),
    )
    return model


def get_config_dataset_info(config: dict) -> tuple:
    """Extract dataset info from config."""
    data_cfg = config.get('data', config)  # Support both flat and nested config
    
    dataset_name = data_cfg.get('dataset', 'cifar10')
    batch_size = data_cfg.get('batch_size', 32)
    num_workers = data_cfg.get('num_workers', 4)
    image_size = data_cfg.get('image_size', 224)
    root = data_cfg.get('root', './data')
    
    return dataset_name, batch_size, num_workers, image_size, root


def get_dataloader(dataset_name, batch_size=32, num_workers=4, image_size=224, root='./data'):
    """Create dataloader for specified dataset."""
    import os
    # Import here to avoid issues if not installed
    from data.cifar import CIFAR10Dataset, CIFAR100Dataset
    from data.rafdb import RAFDBDataset
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    
    if dataset_name.lower() == 'imagefolder':
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        train_root = os.path.join(root, 'train')
        test_root = os.path.join(root, 'test')
        
        # Check if train/test folders exist, otherwise use root
        if not os.path.exists(train_root):
            train_root = root
            test_root = root
        
        train_ds = ImageFolder(train_root, transform=test_transform)
        test_ds = ImageFolder(test_root, transform=test_transform)
        num_classes = len(train_ds.classes)
    elif dataset_name.lower() == 'cifar10':
        train_ds = CIFAR10Dataset(root=root, train=True, download=True)
        test_ds = CIFAR10Dataset(root=root, train=False, download=True)
        num_classes = 10
    elif dataset_name.lower() == 'cifar100':
        train_ds = CIFAR100Dataset(root=root, train=True, download=True)
        test_ds = CIFAR100Dataset(root=root, train=False, download=True)
        num_classes = 100
    elif dataset_name.lower() == 'rafdb':
        train_ds = RAFDBDataset(root=root, split='train', image_size=image_size)
        test_ds = RAFDBDataset(root=root, split='test', image_size=image_size)
        num_classes = 7
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader, num_classes


class HopfieldAnalyzer:
    """Comprehensive analyzer for Hopfield-augmented ViTs."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def get_hopfield_layers(self):
        """Get all Hopfield layers from the model."""
        hopfield_layers = []
        for block in self.model.backbone.blocks:
            if hasattr(block, 'hopfield_layer'):
                hopfield_layers.append(block.hopfield_layer)
        return hopfield_layers
    
    # =========================================================================
    # 1. Memory Pattern Analysis
    # =========================================================================
    
    def analyze_memory_distributions(self):
        """Analyze statistical properties of memory patterns across layers."""
        results = {}
        hopfield_layers = self.get_hopfield_layers()
        
        for layer_idx, hopfield in enumerate(hopfield_layers):
            key_mean = hopfield.memory_keys.mean().item()
            key_std = hopfield.memory_keys.std().item()
            key_norm = hopfield.memory_keys.norm(dim=-1).mean().item()
            
            val_mean = hopfield.memory_vals.mean().item()
            val_std = hopfield.memory_vals.std().item()
            val_norm = hopfield.memory_vals.norm(dim=-1).mean().item()
            
            is_auto = torch.allclose(
                hopfield.memory_keys, hopfield.memory_vals, atol=1e-3
            )
            
            results[f"layer_{layer_idx}"] = {
                "key_mean": key_mean,
                "key_std": key_std,
                "key_norm": key_norm,
                "val_mean": val_mean,
                "val_std": val_std,
                "val_norm": val_norm,
                "is_auto_associative": is_auto,
                "memory_size": hopfield.mem_size,
                "temperature": hopfield.log_temp.exp().item()
            }
        
        return results
    
    def analyze_memory_similarity(self, save_path=None):
        """Analyze pairwise similarity between memory patterns."""
        hopfield_layers = self.get_hopfield_layers()
        
        num_layers = len(hopfield_layers)
        fig, axes = plt.subplots(
            (num_layers + 3) // 4, 4, figsize=(20, 5 * ((num_layers + 3) // 4))
        )
        axes = axes.flatten() if num_layers > 4 else [axes]
        
        entropies = []
        for layer_idx, hopfield in enumerate(hopfield_layers):
            mem_k = hopfield.memory_keys[0].detach().cpu()
            mem_k_norm = F.normalize(mem_k, p=2, dim=-1)
            sim_matrix = torch.mm(mem_k_norm, mem_k_norm.t()).numpy()
            
            ax = axes[layer_idx]
            im = ax.imshow(sim_matrix, cmap='viridis', vmin=-1, vmax=1)
            ax.set_title(f'Layer {layer_idx} Memory Similarity')
            plt.colorbar(im, ax=ax)
            
            entropy = -np.mean(sim_matrix * np.log2(np.clip(sim_matrix, 1e-7, 1)))
            entropies.append(entropy)
        
        for idx in range(len(hopfield_layers), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {"entropies": entropies, "avg_entropy": np.mean(entropies)}
    
    def track_memory_evolution(self, checkpoint_paths, output_file="memory_evolution.npz"):
        """Track how memory patterns evolve during training."""
        all_memories = []
        
        for ckpt_path in tqdm(checkpoint_paths, desc="Loading checkpoints"):
            state_dict = torch.load(ckpt_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            
            layer_memories = []
            for hopfield in self.get_hopfield_layers():
                mem = hopfield.memory_keys[0].detach().cpu()
                layer_memories.append(mem)
            
            all_memories.append(layer_memories)
        
        np.savez(output_file, memories=all_memories)
        
        stability_metrics = []
        for i in range(1, len(all_memories)):
            layer_stability = []
            for layer in range(len(all_memories[0])):
                prev = F.normalize(all_memories[i-1][layer], p=2, dim=-1)
                curr = F.normalize(all_memories[i][layer], p=2, dim=-1)
                sim = torch.diag(torch.mm(prev, curr.t())).mean().item()
                layer_stability.append(sim)
            stability_metrics.append(layer_stability)
        
        plt.figure(figsize=(10, 6))
        stability_metrics = np.array(stability_metrics)
        for layer in range(stability_metrics.shape[1]):
            plt.plot(stability_metrics[:, layer], label=f'Layer {layer}')
        plt.xlabel('Training Step')
        plt.ylabel('Memory Pattern Stability (cosine sim)')
        plt.title('Memory Pattern Evolution During Training')
        plt.legend()
        plt.grid(True)
        plt.savefig('memory_stability.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return {"stability": stability_metrics}
    
    # =========================================================================
    # 2. Query Projection Analysis
    # =========================================================================
    
    def analyze_query_weights(self, dataloader, num_samples=100):
        """Identify which input features most influence memory retrieval."""
        all_grads = []
        
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Analyzing queries")):
            if batch_idx * images.shape[0] >= num_samples:
                break
            
            images = images.to(self.device)
            images.requires_grad = True
            
            with torch.enable_grad():
                features = self.model.backbone.patch_embed(images)
                features = self.model.backbone._pos_embed(features)
                
                block = self.model.backbone.blocks[0]
                hopfield = block.hopfield_layer
                
                x = block.vit_block.norm1(features)
                q = hopfield.to_q(x)
                
                attn = torch.matmul(q, hopfield.memory_keys.transpose(-2, -1)) * hopfield.log_temp.exp()
                attn = F.softmax(attn, dim=-1)
                
                loss = attn.sum()
                loss.backward()
            
            all_grads.append(images.grad.abs().mean(dim=(0, 2, 3)).cpu())
        
        avg_grads = torch.stack(all_grads).mean(0)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.bar(range(3), avg_grads[:3].numpy())
        plt.xticks(range(3), ['R', 'G', 'B'])
        plt.title('Channel Importance')
        
        patch_importance = avg_grads[3:].numpy()
        if len(patch_importance) > 0:
            num_patches = int(np.sqrt(len(patch_importance)))
            patch_importance = patch_importance[:num_patches**2].reshape(num_patches, num_patches)
            plt.subplot(1, 2, 2)
            plt.imshow(patch_importance, cmap='hot')
            plt.colorbar()
            plt.title('Spatial Importance Map')
        
        plt.tight_layout()
        plt.savefig('query_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return {"avg_grads": avg_grads.numpy()}
    
    def analyze_query_subspace(self, dataloader, num_samples=500):
        """Analyze the subspace spanned by query projections."""
        all_queries = []
        
        self.model.eval()
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Collecting queries"):
                images = images.to(self.device)
                features = self.model.backbone.patch_embed(images)
                features = self.model.backbone._pos_embed(features)
                
                block = self.model.backbone.blocks[0]
                hopfield = block.hopfield_layer
                x = block.vit_block.norm1(features)
                q = hopfield.to_q(x)
                
                all_queries.append(q[:, 0].detach().cpu())
                
                if sum(x.shape[0] for x in all_queries) >= num_samples:
                    break
        
        queries = torch.cat(all_queries, dim=0)[:num_samples]
        
        pca = PCA(n_components=min(50, num_samples))
        reduced = pca.fit_transform(queries.numpy())
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Query Vector Dimensionality')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5, s=10)
        plt.title('Query Vectors in PCA Space')
        
        plt.tight_layout()
        plt.savefig('query_subspace.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        threshold = 0.9
        dims_needed = np.where(np.cumsum(pca.explained_variance_ratio_) >= threshold)[0][0] + 1
        
        return {
            "explained_variance": pca.explained_variance_ratio_,
            "dims_needed_90": dims_needed,
            "reduced": reduced
        }
    
    # =========================================================================
    # 3. Temperature Parameter Analysis
    # =========================================================================
    
    def monitor_temperature_dynamics(self, dataloader):
        """Track how temperature affects retrieval during training."""
        temperature_history = []
        entropy_history = []
        
        self.model.eval()
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Monitoring temperature"):
                images = images.to(self.device)
                
                for hopfield in self.get_hopfield_layers():
                    features = self.model.backbone.patch_embed(images)
                    features = self.model.backbone._pos_embed(features)
                    
                    x = hopfield.norm(features) if hasattr(hopfield, 'norm') else features
                    q = hopfield.to_q(x)
                    
                    attn = torch.matmul(q, hopfield.memory_keys.transpose(-2, -1)) * hopfield.log_temp.exp()
                    attn = F.softmax(attn, dim=-1)
                    
                    entropy = -torch.sum(attn * torch.log(attn + 1e-7), dim=-1).mean().item()
                    
                    temperature_history.append(hopfield.log_temp.exp().item())
                    entropy_history.append(entropy)
                
                break
        
        plt.figure(figsize=(10, 5))
        plt.scatter(temperature_history, entropy_history, alpha=0.6)
        plt.xlabel('Temperature Parameter')
        plt.ylabel('Attention Entropy')
        plt.title('Temperature vs. Retrieval Entropy')
        plt.grid(True)
        plt.savefig('temperature_entropy.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        corr = np.corrcoef(temperature_history, entropy_history)[0, 1] if len(temperature_history) > 1 else 0
        
        return {
            "temperatures": temperature_history,
            "entropies": entropy_history,
            "correlation": corr
        }
    
    def analyze_temperature_sensitivity(self, dataloader, layer_idx=0, num_samples=10):
        """Test how model predictions change with temperature."""
        images, _ = next(iter(dataloader))
        images = images.to(self.device)[:num_samples]
        
        original_temp = self.model.backbone.blocks[layer_idx].hopfield_layer.log_temp.data.clone()
        temperatures = np.logspace(-2, 1, 20)
        
        predictions = []
        for temp in tqdm(temperatures, desc="Testing temperatures"):
            self.model.backbone.blocks[layer_idx].hopfield_layer.log_temp.data = torch.log(torch.tensor([temp]))
            
            with torch.no_grad():
                logits = self.model(images)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs.cpu())
        
        self.model.backbone.blocks[layer_idx].hopfield_layer.log_temp.data = original_temp
        
        predictions = torch.stack(predictions)
        stability = F.cosine_similarity(
            predictions[0].unsqueeze(0), 
            predictions, 
            dim=-1
        ).mean(dim=1).numpy()
        
        plt.figure(figsize=(10, 5))
        plt.semilogx(temperatures, stability)
        plt.xlabel('Temperature (log scale)')
        plt.ylabel('Prediction Stability')
        plt.title('Model Sensitivity to Temperature Parameter')
        plt.grid(True, which="both", ls="-")
        plt.savefig('temperature_sensitivity.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        stable_idx = np.where(stability > 0.95)[0]
        stable_range = (temperatures[stable_idx[0]], temperatures[stable_idx[-1]]) if len(stable_idx) > 0 else None
        
        return {
            "temperatures": temperatures,
            "stability": stability,
            "stable_range": stable_range
        }
    
    # =========================================================================
    # 4. Memory Utilization Analysis
    # =========================================================================
    
    def track_memory_utilization(self, dataloader, num_batches=10):
        """Track which memory slots get activated across different inputs."""
        activation_counts = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Tracking utilization")):
                if batch_idx >= num_batches:
                    break
                
                images = images.to(self.device)
                features = self.model.backbone.patch_embed(images)
                features = self.model.backbone._pos_embed(features)
                
                batch_activations = []
                hopfield_layers = self.get_hopfield_layers()
                
                for block_idx, block in enumerate(self.model.backbone.blocks):
                    if not hasattr(block, 'hopfield_layer'):
                        continue
                    
                    x = features
                    for prev_block in self.model.backbone.blocks[:block_idx]:
                        x = prev_block(x)
                    
                    hopfield = block.hopfield_layer
                    x_norm = hopfield.norm(x) if hasattr(hopfield, 'norm') else x
                    q = hopfield.to_q(x_norm)
                    
                    attn = torch.matmul(q, hopfield.memory_keys.transpose(-2, -1)) * hopfield.log_temp.exp()
                    attn = F.softmax(attn, dim=-1)
                    
                    _, top_indices = torch.topk(attn, k=5, dim=-1)
                    activations = torch.zeros(hopfield.mem_size, device='cpu')
                    for idx in top_indices.view(-1):
                        activations[idx] += 1
                    
                    batch_activations.append(activations.cpu())
                
                activation_counts.append(batch_activations)
        
        total_activations = []
        for layer_idx in range(len(activation_counts[0])):
            layer_acts = torch.stack([batch[layer_idx] for batch in activation_counts]).sum(0)
            total_activations.append(layer_acts)
        
        num_layers = len(total_activations)
        fig, axes = plt.subplots(
            (num_layers + 3) // 4, 4, figsize=(20, 5 * ((num_layers + 3) // 4))
        )
        axes = axes.flatten() if num_layers > 4 else [axes]
        
        utilization_metrics = []
        for layer_idx, acts in enumerate(total_activations):
            axes[layer_idx].hist(acts.numpy(), bins=50)
            axes[layer_idx].set_title(f'Layer {layer_idx} Memory Utilization')
            axes[layer_idx].set_xlabel('Activation Count')
            axes[layer_idx].set_ylabel('Memory Slots')
            
            utilization = (acts > 0).float().mean().item()
            entropy = -torch.sum(acts * torch.log(acts + 1e-7)) / acts.sum()
            utilization_metrics.append({
                "utilization": utilization,
                "entropy": entropy.item()
            })
        
        for idx in range(len(total_activations), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('memory_utilization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return {"metrics": utilization_metrics}
    
    def analyze_memory_specialization(self, dataloader, num_classes):
        """Determine if memory slots specialize for specific classes."""
        class_activations = {cls: [] for cls in range(num_classes)}
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Analyzing specialization"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                features = self.model.backbone.patch_embed(images)
                features = self.model.backbone._pos_embed(features)
                
                for block in self.model.backbone.blocks:
                    if hasattr(block, 'hopfield_layer'):
                        hopfield = block.hopfield_layer
                        x = hopfield.norm(features) if hasattr(hopfield, 'norm') else features
                        q = hopfield.to_q(x)
                        
                        attn = torch.matmul(q, hopfield.memory_keys.transpose(-2, -1)) * hopfield.log_temp.exp()
                        attn = F.softmax(attn, dim=-1)
                        
                        for i in range(labels.shape[0]):
                            cls_idx = labels[i].item()
                            top_slot = torch.argmax(attn[i, 0]).item()
                            class_activations[cls_idx].append(top_slot)
                
                break
        
        specialization = []
        mem_size = self.get_hopfield_layers()[0].mem_size if self.get_hopfield_layers() else 1000
        
        for cls_idx, activations in class_activations.items():
            if not activations:
                continue
            
            slot_counts = torch.bincount(
                torch.tensor(activations), 
                minlength=mem_size
            ).float()
            
            probs = slot_counts / slot_counts.sum()
            entropy = -torch.sum(probs * torch.log(probs + 1e-7))
            max_entropy = np.log(mem_size)
            specialization_score = 1.0 - (entropy / max_entropy)
            
            specialization.append((cls_idx, specialization_score.item()))
        
        specialization.sort(key=lambda x: x[1], reverse=True)
        
        plt.figure(figsize=(12, 6))
        top_classes = specialization[:10]
        plt.bar(range(len(top_classes)), [s for _, s in top_classes])
        plt.xticks(range(len(top_classes)), [f'Class {c}' for c, _ in top_classes], rotation=45)
        plt.title('Memory Specialization by Class')
        plt.ylabel('Specialization Score (0-1)')
        plt.tight_layout()
        plt.savefig('memory_specialization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return {"specialization": specialization}
    
    # =========================================================================
    # 5. Cross-Component Analysis
    # =========================================================================
    
    def compare_attention_patterns(self, dataloader, layer_idx=3):
        """Compare standard ViT attention with Hopfield retrieval patterns."""
        images, _ = next(iter(dataloader))
        images = images.to(self.device)[:4]
        
        original_block = self.model.backbone.blocks[layer_idx]
        original_hopfield = original_block.hopfield_layer
        
        class NoHopfieldBlock(nn.Module):
            def __init__(self, vit_block):
                super().__init__()
                self.vit_block = vit_block
                
            def forward(self, x):
                x = self.vit_block.norm1(x)
                x = self.vit_block.attn(x)
                x = x + self.vit_block.drop_path(self.vit_block.mlp(self.vit_block.norm2(x)))
                return x
        
        self.model.backbone.blocks[layer_idx] = NoHopfieldBlock(original_block.vit_block)
        
        self.model.eval()
        with torch.no_grad():
            features = self.model.backbone.patch_embed(images)
            features = self.model.backbone._pos_embed(features)
            
            for i in range(layer_idx):
                features = self.model.backbone.blocks[i](features)
            
            x = original_block.vit_block.norm1(features)
            attn_vit = original_block.vit_block.attn.get_attention(x)
        
        self.model.backbone.blocks[layer_idx] = original_block
        
        with torch.no_grad():
            x = original_block.vit_block.norm1(features)
            q = original_hopfield.to_q(x)
            attn_hopfield = F.softmax(
                torch.matmul(q, original_hopfield.memory_keys.transpose(-2, -1)) * 
                original_hopfield.log_temp.exp(), 
                dim=-1
            )
        
        num_heads = attn_vit.shape[1] if len(attn_vit.shape) > 3 else 1
        num_patches = int(np.sqrt(attn_hopfield.shape[1]))
        
        fig, axes = plt.subplots(3, min(num_heads, 4), figsize=(15, 12))
        if num_heads == 1:
            axes = axes.T
        
        for head in range(min(num_heads, 4)):
            axes[0, head].imshow(attn_vit[0, head].cpu().numpy())
            axes[0, head].set_title(f'ViT Attention Head {head}')
            
            axes[1, head].imshow(attn_hopfield[0, head].cpu().numpy().reshape(num_patches, num_patches), cmap='hot')
            axes[1, head].set_title(f'Hopfield Retrieval Head {head}')
            
            diff = attn_vit[0, head].cpu().numpy().flatten() - attn_hopfield[0, head].cpu().numpy()
            diff = diff[:num_patches**2].reshape(num_patches, num_patches)
            im = axes[2, head].imshow(diff, cmap='coolwarm', vmin=-0.1, vmax=0.1)
            axes[2, head].set_title(f'Difference Head {head}')
            plt.colorbar(im, ax=axes[2, head])
        
        plt.tight_layout()
        plt.savefig('attention_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        hopfield_flat = attn_hopfield[0, 0].cpu().numpy().flatten()[:attn_vit[0, 0].numel()]
        cosine_sim = F.cosine_similarity(
            torch.tensor(attn_vit[0, 0].flatten()), 
            torch.tensor(hopfield_flat), 
            dim=0
        ).item()
        
        return {"cosine_similarity": cosine_sim}
    
    def analyze_feature_alignment(self, dataloader, num_samples=500):
        """Analyze how Hopfield layers transform the feature space."""
        from copy import deepcopy
        
        features_no_hopfield = []
        features_with_hopfield = []
        labels_list = []
        
        original_blocks = [block for block in self.model.backbone.blocks]
        
        class NoHopfieldBlock(nn.Module):
            def __init__(self, vit_block):
                super().__init__()
                self.vit_block = vit_block
                
            def forward(self, x):
                x = self.vit_block.norm1(x)
                x = self.vit_block.attn(x)
                x = x + self.vit_block.drop_path(self.vit_block.mlp(self.vit_block.norm2(x)))
                return x
        
        no_hopfield_blocks = nn.ModuleList([
            NoHopfieldBlock(block.vit_block) if hasattr(block, 'hopfield_layer') else block
            for block in original_blocks
        ])
        
        self.model.backbone.blocks = no_hopfield_blocks
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting features (no hopfield)"):
                images = images.to(self.device)
                features = self.model.backbone.forward_features(images)
                if isinstance(features, tuple):
                    features = features[0]
                cls_token = features[:, 0]
                features_no_hopfield.append(cls_token.cpu())
                labels_list.append(labels.cpu())
                
                if sum(x.shape[0] for x in features_no_hopfield) >= num_samples:
                    break
        
        self.model.backbone.blocks = nn.ModuleList(original_blocks)
        
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Extracting features (with hopfield)"):
                images = images.to(self.device)
                features = self.model.backbone.forward_features(images)
                if isinstance(features, tuple):
                    features = features[0]
                cls_token = features[:, 0]
                features_with_hopfield.append(cls_token.cpu())
                
                if sum(x.shape[0] for x in features_with_hopfield) >= num_samples:
                    break
        
        features_no_hopfield = torch.cat(features_no_hopfield, dim=0)[:num_samples]
        features_with_hopfield = torch.cat(features_with_hopfield, dim=0)[:num_samples]
        labels = torch.cat(labels_list, dim=0)[:num_samples]
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(features_no_hopfield.numpy(), labels.numpy())
        acc_no_hopfield = knn.score(features_no_hopfield.numpy(), labels.numpy())
        acc_transfer = knn.score(features_with_hopfield.numpy(), labels.numpy())
        
        combined = torch.cat([features_no_hopfield, features_with_hopfield], dim=0)
        tsne = TSNE(n_components=2, random_state=42)
        embedded = tsne.fit_transform(combined.numpy())
        
        plt.figure(figsize=(12, 6))
        plt.scatter(embedded[:len(features_no_hopfield), 0], 
                    embedded[:len(features_no_hopfield), 1],
                    c=labels.numpy(), cmap='tab10', alpha=0.5, s=10, label='No Hopfield')
        plt.scatter(embedded[len(features_no_hopfield):, 0], 
                    embedded[len(features_no_hopfield):, 1],
                    c=labels.numpy(), cmap='tab10', alpha=0.5, s=10, marker='x', label='With Hopfield')
        plt.legend()
        plt.title('Feature Space Alignment (t-SNE)')
        plt.savefig('feature_alignment.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            "knn_accuracy_no_hopfield": acc_no_hopfield,
            "knn_accuracy_transfer": acc_transfer,
            "preservation_ratio": acc_transfer / acc_no_hopfield
        }
    
    # =========================================================================
    # 6. Advanced Diagnostics
    # =========================================================================
    
    def visualize_memory_patterns(self, dataset, num_patterns=16, layer_idx=6):
        """Visualize what memory patterns 'look like' in image space."""
        hopfield = self.model.backbone.blocks[layer_idx].hopfield_layer
        
        rand_idx = torch.randperm(len(dataset))[:100]
        rand_images = torch.stack([dataset[i][0] for i in rand_idx]).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            rand_features = self.model.backbone.patch_embed(rand_images)
            rand_features = self.model.backbone._pos_embed(rand_features)
            for block in self.model.backbone.blocks[:layer_idx]:
                rand_features = block(rand_features)
            rand_features = self.model.backbone.blocks[layer_idx].vit_block.norm1(rand_features)
        
        max_queries = []
        for slot_idx in range(min(hopfield.mem_size, num_patterns)):
            target_attn = torch.zeros(hopfield.mem_size, device=self.device)
            target_attn[slot_idx] = 1.0
            
            k = F.normalize(hopfield.memory_keys[0], p=2, dim=-1)
            q = target_attn @ k
            
            max_queries.append(q)
        
        max_queries = torch.stack(max_queries)
        
        best_matches = []
        for q in max_queries:
            rand_queries = hopfield.to_q(rand_features)
            
            sims = F.cosine_similarity(
                rand_queries.reshape(-1, rand_queries.shape[-1]), 
                q.unsqueeze(0),
                dim=-1
            )
            _, idx = torch.topk(sims, k=1)
            best_matches.append(rand_idx[idx // rand_features.shape[1]])
        
        num_cols = 4
        num_rows = (num_patterns + num_cols - 1) // num_cols
        plt.figure(figsize=(15, 3 * num_rows))
        for i in range(num_patterns):
            img, _ = dataset[best_matches[i]]
            plt.subplot(num_rows, num_cols, i+1)
            plt.imshow(img.permute(1, 2, 0).numpy())
            plt.title(f'Memory {i}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('memory_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return {"best_matches": best_matches}
    
    def run_full_analysis(self, dataloader, dataset=None, output_dir="analysis_output"):
        """Run all analysis methods and save results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {}
        
        print("\n=== 1. Memory Pattern Analysis ===")
        results['memory_dist'] = self.analyze_memory_distributions()
        print(f"Memory distributions: {results['memory_dist']}")
        
        results['memory_similarity'] = self.analyze_memory_similarity(
            save_path=output_path / "memory_similarity.png"
        )
        print(f"Memory similarity entropy: {results['memory_similarity']['avg_entropy']:.4f}")
        
        print("\n=== 2. Query Projection Analysis ===")
        results['query_importance'] = self.analyze_query_weights(dataloader)
        
        results['query_subspace'] = self.analyze_query_subspace(dataloader)
        print(f"Query subspace dims (90% var): {results['query_subspace']['dims_needed_90']}")
        
        print("\n=== 3. Temperature Analysis ===")
        results['temp_dynamics'] = self.monitor_temperature_dynamics(dataloader)
        print(f"Temperature-entropy correlation: {results['temp_dynamics']['correlation']:.4f}")
        
        results['temp_sensitivity'] = self.analyze_temperature_sensitivity(dataloader)
        if results['temp_sensitivity']['stable_range']:
            print(f"Stable temperature range: {results['temp_sensitivity']['stable_range']}")
        
        print("\n=== 4. Memory Utilization ===")
        results['utilization'] = self.track_memory_utilization(dataloader)
        for i, m in enumerate(results['utilization']['metrics']):
            print(f"Layer {i}: utilization={m['utilization']:.2%}, entropy={m['entropy']:.4f}")
        
        print("\n=== 5. Attention Comparison ===")
        results['attention_comp'] = self.compare_attention_patterns(dataloader)
        print(f"Attention similarity: {results['attention_comp']['cosine_similarity']:.4f}")
        
        results['feature_alignment'] = self.analyze_feature_alignment(dataloader)
        print(f"Feature alignment: {results['feature_alignment']['preservation_ratio']:.4f}")
        
        if dataset is not None:
            print("\n=== 6. Advanced Diagnostics ===")
            try:
                results['memory_viz'] = self.visualize_memory_patterns(dataset)
            except Exception as e:
                print(f"Memory visualization skipped: {e}")
        
        print(f"\n=== Analysis complete! Results saved to {output_dir} ===")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Hopfield ViT Analyzer")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config (optional, auto-detects from checkpoint dir if not provided)')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--output_dir', type=str, default='analysis_output', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    # Try to load config from same directory as checkpoint if not provided
    config = None
    if args.config:
        print(f"Loading config from {args.config}...")
        config = load_config(args.config)
    else:
        # Try to find config in checkpoint directory
        checkpoint_dir = Path(args.checkpoint).parent
        possible_configs = [
            checkpoint_dir / 'config.yaml',
            checkpoint_dir / 'config.yml',
            checkpoint_dir / '..' / 'config.yaml',
        ]
        for cfg_path in possible_configs:
            if cfg_path.exists():
                print(f"Auto-detected config: {cfg_path}")
                config = load_config(str(cfg_path))
                break
    
    # Build model config
    if config:
        model_config = config.get('model', config)
        num_classes = args.dataset or get_config_dataset_info(config)[0]  # Will be overridden by dataset arg
    else:
        model_config = {}
        num_classes = 10
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {args.checkpoint}...")
    model = get_model(config or {'model': model_config, 'num_classes': num_classes})
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    
    # Get dataset info from config or args
    if config:
        dataset_name, batch_size, num_workers, image_size, root = get_config_dataset_info(config)
        if args.dataset:
            dataset_name = args.dataset
        if args.batch_size:
            batch_size = args.batch_size
    else:
        dataset_name = args.dataset or 'cifar10'
        batch_size = args.batch_size or 32
        num_workers = 4
        image_size = 224
        root = './data'
    
    print(f"Loading dataset: {dataset_name}...")
    train_loader, test_loader, num_classes = get_dataloader(
        dataset_name, batch_size, num_workers, image_size, root
    )
    
    analyzer = HopfieldAnalyzer(model, device=device)
    results = analyzer.run_full_analysis(train_loader, dataset=train_loader.dataset, output_dir=args.output_dir)
    
    print("\n=== Summary ===")
    print(f"Memory pattern entropy: {results['memory_similarity']['avg_entropy']:.4f}")
    print(f"Query subspace dimensionality: {results['query_subspace']['dims_needed_90']}")
    print(f"Temperature-entropy correlation: {results['temp_dynamics']['correlation']:.4f}")
    print(f"Attention pattern similarity: {results['attention_comp']['cosine_similarity']:.4f}")


if __name__ == "__main__":
    main()
