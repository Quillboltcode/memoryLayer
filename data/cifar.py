import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from typing import Optional, Tuple, Callable, Any
import os
from pathlib import Path


class CIFAR10Dataset:
    """CIFAR-10 dataset wrapper with transforms."""
    
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        image_size: int = 224,
        augment: bool = True,
        download: bool = True,
    ):
        self.root = root
        self.train = train
        self.image_size = image_size
        
        if augment and train:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return {'image': img, 'label': label}
    
    def get_dataloader(
        self, 
        batch_size: int = 64, 
        shuffle: bool = True, 
        num_workers: int = 4,
        **kwargs
    ) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )


class CIFAR100Dataset:
    """CIFAR-100 dataset wrapper with transforms."""
    
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        image_size: int = 224,
        augment: bool = True,
        download: bool = True,
    ):
        self.root = root
        self.train = train
        self.image_size = image_size
        
        if augment and train:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        self.dataset = datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return {'image': img, 'label': label}
    
    def get_dataloader(
        self, 
        batch_size: int = 64, 
        shuffle: bool = True, 
        num_workers: int = 4,
        **kwargs
    ) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )


class FewShotCIFAR10:
    """Few-shot CIFAR-10 for limited data experiments."""
    
    def __init__(
        self,
        root: str = "./data",
        n_way: int = 10,
        n_shot: int = 5,
        image_size: int = 224,
        augment: bool = False,
    ):
        from collections import defaultdict
        
        self.root = root
        self.n_way = n_way
        self.n_shot = n_shot
        self.image_size = image_size
        
        full_dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        )
        
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(full_dataset):
            class_indices[label].append(idx)
        
        selected_indices = []
        for c in range(n_way):
            selected_indices.extend(class_indices[c][:n_shot])
        
        self.indices = selected_indices
        self.dataset = full_dataset
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]
        return {'image': img, 'label': label}
    
    def get_dataloader(self, batch_size: int = 64, shuffle: bool = True, **kwargs):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )


class FewShotCIFAR100:
    """Few-shot CIFAR-100 for limited data experiments."""
    
    def __init__(
        self,
        root: str = "./data",
        n_way: int = 100,
        n_shot: int = 10,
        image_size: int = 224,
        augment: bool = False,
    ):
        from collections import defaultdict
        
        self.root = root
        self.n_way = n_way
        self.n_shot = n_shot
        self.image_size = image_size
        
        full_dataset = datasets.CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        )
        
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(full_dataset):
            class_indices[label].append(idx)
        
        selected_indices = []
        for c in range(n_way):
            selected_indices.extend(class_indices[c][:n_shot])
        
        self.indices = selected_indices
        self.dataset = full_dataset
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]
        return {'image': img, 'label': label}
    
    def get_dataloader(self, batch_size: int = 64, shuffle: bool = True, **kwargs):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )


def get_cifar10_dataloaders(
    root: str = "./data",
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 train and val dataloaders."""
    train_dataset = CIFAR10Dataset(
        root=root,
        train=True,
        image_size=image_size,
        augment=True,
    )
    
    val_dataset = CIFAR10Dataset(
        root=root,
        train=False,
        image_size=image_size,
        augment=False,
    )
    
    train_loader = train_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = val_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader


def get_cifar100_dataloaders(
    root: str = "./data",
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Get CIFAR-100 train and val dataloaders."""
    train_dataset = CIFAR100Dataset(
        root=root,
        train=True,
        image_size=image_size,
        augment=True,
    )
    
    val_dataset = CIFAR100Dataset(
        root=root,
        train=False,
        image_size=image_size,
        augment=False,
    )
    
    train_loader = train_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = val_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader
