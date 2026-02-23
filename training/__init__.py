"""
Training utilities for Memory-VLM.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, List, Union
from tqdm import tqdm


class Trainer:
    """Trainer class for MemoryCLIP model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda',
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.config = config or {}
        self.logger = logger
        
        self.model.to(self.device)
        
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 5e-4),
                weight_decay=self.config.get('weight_decay', 0.01),
            )
        else:
            self.optimizer = optimizer
        
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        self.global_step = 0
        self.current_epoch = 0
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch in pbar:
            images = batch['image'].to(self.device)
            texts = batch['text']
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with autocast():
                    loss = self._compute_loss(images, texts)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self._compute_loss(images, texts)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if self.logger and self.global_step % 10 == 0:
                self.logger.log({'train_loss': loss.item(), 'step': self.global_step})
            
            self.global_step += 1
        
        return total_loss / num_batches
    
    def _compute_loss(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """Compute contrastive loss."""
        return self.model.contrastive_loss(images, texts)
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            texts = batch['text']
            
            if self.scaler is not None:
                with autocast():
                    loss = self._compute_loss(images, texts)
            else:
                loss = self._compute_loss(images, texts)
            
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(
        self,
        num_epochs: int,
        save_interval: int = 1,
        output_dir: str = 'checkpoints',
    ):
        """Run full training loop."""
        os.makedirs(output_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            train_loss = self.train_epoch()
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            if self.logger:
                self.logger.log({'train_epoch_loss': train_loss, 'epoch': epoch})
            
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Epoch {epoch}/{num_epochs} - Val Loss: {val_loss:.4f}")
                
                if self.logger:
                    self.logger.log({'val_loss': val_loss, 'epoch': epoch})
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(os.path.join(output_dir, 'best.pt'))
            
            if epoch % save_interval == 0:
                self.save_checkpoint(os.path.join(output_dir, f'epoch_{epoch}.pt'))
        
        self.save_checkpoint(os.path.join(output_dir, 'last.pt'))
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'config': self.config,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']


class ClassificationTrainer:
    """Trainer class for ViT-based classification models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda',
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.config = config or {}
        self.logger = logger
        
        self.model.to(self.device)
        
        if optimizer is None:
            lr = self.config.get('learning_rate', 1e-3)
            weight_decay = self.config.get('weight_decay', 0.01)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optimizer
        
        self.use_amp = self.config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        self.scheduler = None
        scheduler_config = self.config.get('scheduler', 'none')
        if scheduler_config == 'cosine':
            epochs = self.config.get('epochs', 100)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        
        self.global_step = 0
        self.current_epoch = 0
        self.best_acc = 0.0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = nn.functional.cross_entropy(outputs, labels)
                
                self.scaler.scale(loss).backward()
                if self.config.get('clip_grad_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['clip_grad_norm']
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                if self.config.get('clip_grad_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['clip_grad_norm']
                    )
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
            
            if self.logger and self.global_step % 10 == 0:
                self.logger.log({
                    'train_loss': loss.item(), 
                    'train_acc': 100. * correct / total,
                    'step': self.global_step
                })
            
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    @torch.no_grad()
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate the model on validation or test set."""
        if dataloader is None:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = len(dataloader)
        
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = nn.functional.cross_entropy(outputs, labels)
            else:
                outputs = self.model(images)
                loss = nn.functional.cross_entropy(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / num_batches
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(
        self,
        num_epochs: int,
        save_interval: int = 1,
        output_dir: str = 'checkpoints',
        eval_on_test: bool = True,
    ) -> Dict[str, Any]:
        """Run full training loop."""
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'train_losses': [],
            'train_accs': [],
            'val_losses': [],
            'val_accs': [],
            'test_acc': 0.0,
        }
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            
            results['train_losses'].append(train_metrics['loss'])
            results['train_accs'].append(train_metrics['accuracy'])
            
            if self.logger:
                self.logger.log({
                    'train_loss': train_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'epoch': epoch
                })
            
            if self.val_loader is not None and epoch % self.config.get('eval_interval', 1) == 0:
                val_metrics = self.evaluate(self.val_loader)
                print(f"Epoch {epoch}/{num_epochs} - Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
                
                results['val_losses'].append(val_metrics['loss'])
                results['val_accs'].append(val_metrics['accuracy'])
                
                if self.logger:
                    self.logger.log({
                        'val_loss': val_metrics['loss'],
                        'val_acc': val_metrics['accuracy'],
                        'epoch': epoch
                    })
                
                if val_metrics['accuracy'] > self.best_acc:
                    self.best_acc = val_metrics['accuracy']
                    self.save_checkpoint(os.path.join(output_dir, 'best.pt'))
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            if epoch % save_interval == 0:
                self.save_checkpoint(os.path.join(output_dir, f'epoch_{epoch}.pt'))
        
        self.save_checkpoint(os.path.join(output_dir, 'last.pt'))
        
        if eval_on_test and self.test_loader is not None:
            test_metrics = self.evaluate(self.test_loader)
            print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
            results['test_acc'] = test_metrics['accuracy']
            
            if self.logger:
                self.logger.log({'test_acc': test_metrics['accuracy']})
        
        return results
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'best_acc': self.best_acc,
            'config': self.config,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']
        self.best_acc = checkpoint.get('best_acc', 0.0)


__all__ = ['Trainer', 'ClassificationTrainer']
