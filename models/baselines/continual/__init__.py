import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any


class EWC:
    """
    Elastic Weight Consolidation (EWC) for continual learning.
    
    Reference: Kirkpatrick et al. 2017 - "Overcoming catastrophic forgetting in neural networks"
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataloader: Optional[Any] = None,
        ewc_lambda: float = 1000.0,
        device: str = "cuda",
    ):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.device = device
        
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        
        self._compute_fisher(dataloader)
    
    def _compute_fisher(self, dataloader):
        """Compute Fisher information matrix."""
        if dataloader is None:
            return
        
        self.model.eval()
        for batch in dataloader:
            if isinstance(batch, dict):
                inputs = batch.get('image', batch.get('img', None))
                labels = batch.get('label', batch.get('target', None))
            else:
                inputs, labels = batch
            
            if inputs is None or labels is None:
                continue
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += p.grad.data.clone() ** 2
    
        for n in self.fisher:
            self.fisher[n] /= len(dataloader) if hasattr(dataloader, '__len__') else 1
    
    def penalty(self) -> torch.Tensor:
        """Compute EWC penalty term."""
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.params:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return loss * self.ewc_lambda
    
    def update(self, dataloader):
        """Update stored parameters and Fisher information."""
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self._compute_fisher(dataloader)


class LwF:
    """
    Learning without Forgetting (LwF) for continual learning.
    
    Reference: Li & Hoiem 2016 - "Learning without Forgetting"
    """
    
    def __init__(
        self,
        model: nn.Module,
        old_model: Optional[nn.Module] = None,
        alpha: float = 1.0,
        temperature: float = 2.0,
        device: str = "cuda",
    ):
        self.model = model
        self.old_model = old_model
        self.alpha = alpha
        self.temperature = temperature
        self.device = device
        
        if old_model is not None:
            self.old_model.eval()
            for param in self.old_model.parameters():
                param.requires_grad = False
    
    def distillation_loss(
        self, 
        new_outputs: torch.Tensor, 
        old_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Compute knowledge distillation loss."""
        log_probs = torch.log_softmax(new_outputs / self.temperature, dim=-1)
        old_probs = torch.softmax(old_outputs / self.temperature, dim=-1)
        
        loss = -torch.sum(old_probs * log_probs, dim=-1).mean()
        return loss * (self.temperature ** 2)
    
    def forward(self, inputs: torch.Tensor) -> tuple:
        """Forward pass with knowledge distillation."""
        new_outputs = self.model(inputs)
        
        if self.old_model is not None:
            with torch.no_grad():
                old_outputs = self.old_model(inputs)
            return new_outputs, old_outputs
        
        return new_outputs, None


class iCaRL:
    """
    iCaRL: Incremental Classifier and Representation Learning.
    
    Reference: Rebuffi et al. 2017 - "iCaRL: Incremental Classifier and Representation Learning"
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 100,
        memory_size: int = 2000,
        device: str = "cuda",
    ):
        self.model = model
        self.num_classes = num_classes
        self.memory_size = memory_size
        self.device = device
        
        self.class_means: Dict[int, torch.Tensor] = {}
        self.exemplars: Dict[int, List[torch.Tensor]] = {}
        
        self.old_head: Optional[nn.Linear] = None
    
    def compute_class_means(self, dataloader):
        """Compute mean representation for each class."""
        self.model.eval()
        
        class_features: Dict[int, List[torch.Tensor]] = {}
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    inputs = batch.get('image', batch.get('img', None))
                    labels = batch.get('label', batch.get('target', None))
                else:
                    inputs, labels = batch
                
                if inputs is None or labels is None:
                    continue
                
                inputs = inputs.to(self.device)
                
                features = self.model.extract_features(inputs)
                if isinstance(features, tuple):
                    features = features[0]
                
                features = features.cpu()
                
                for i, label in enumerate(labels):
                    label = label.item() if label.dim() == 0 else label
                    if label not in class_features:
                        class_features[label] = []
                    class_features[label].append(features[i])
        
        for label, feat_list in class_features.items():
            feats = torch.stack(feat_list)
            self.class_means[label] = feats.mean(dim=0)
    
    def reduce_exemplars(self, m: int):
        """Reduce exemplars to m per class."""
        for label in self.exemplars:
            if len(self.exemplars[label]) > m:
                self.exemplars[label] = self.exemplars[label][:m]
    
    def construct_exemplars(self, dataloader, samples_per_class: int = 20):
        """Construct exemplar sets for each class."""
        self.model.eval()
        
        class_features: Dict[int, List[tuple]] = {}
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    inputs = batch.get('image', batch.get('img', None))
                    labels = batch.get('label', batch.get('target', None))
                else:
                    inputs, labels = batch
                
                if inputs is None or labels is None:
                    continue
                
                inputs = inputs.to(self.device)
                
                features = self.model.extract_features(inputs)
                if isinstance(features, tuple):
                    features = features[0]
                
                for i, label in enumerate(labels):
                    label = label.item() if label.dim() == 0 else label
                    if label not in class_features:
                        class_features[label] = []
                    class_features[label].append((features[i].cpu(), inputs[i].cpu()))
        
        n_per_class = min(samples_per_class, self.memory_size // len(class_features))
        
        for label, feat_img_pairs in class_features.items():
            feat_img_pairs.sort(key=lambda x: x[0].norm(), reverse=True)
            
            self.exemplars[label] = [img for _, img in feat_img_pairs[:n_per_class]]
    
    def classify(self, inputs: torch.Tensor) -> torch.Tensor:
        """Classify using nearest-mean-of-exemplars."""
        self.model.eval()
        
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            features = self.model.extract_features(inputs)
            if isinstance(features, tuple):
                features = features[0]
        
        logits = torch.zeros(inputs.size(0), self.num_classes, device=self.device)
        
        for label, mean in self.class_means.items():
            mean = mean.to(self.device)
            dist = (features - mean).norm(dim=-1)
            logits[:, label] = -dist
        
        return logits.argmax(dim=-1)


class ReplayBuffer:
    """
    Simple experience replay buffer for continual learning.
    """
    
    def __init__(self, capacity: int = 1000, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self.buffer: List[Dict] = []
    
    def add(self, batch: Dict):
        """Add batch to buffer."""
        for item in batch:
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)
            self.buffer.append({k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in item.items()})
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample random batch from buffer."""
        if len(self.buffer) == 0:
            return []
        
        indices = torch.randint(0, len(self.buffer), (min(batch_size, len(self.buffer)),))
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


def create_ewc_plugin(model, dataloader=None, ewc_lambda=1000):
    """Factory function to create EWC plugin."""
    return EWC(model, dataloader, ewc_lambda)


def create_lwf_plugin(model, old_model=None, alpha=1.0):
    """Factory function to create LwF plugin."""
    return LwF(model, old_model, alpha)


def create_icarl_plugin(model, num_classes=100, memory_size=2000):
    """Factory function to create iCaRL plugin."""
    return iCaRL(model, num_classes, memory_size)
