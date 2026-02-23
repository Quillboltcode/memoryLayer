import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, Tuple
import os
from pathlib import Path


class RAFDBDataset(Dataset):
    """
    RAFDB (Real-World Affective Faces Database) dataset.
    
    Note: This dataset requires manual download from:
    http://www.whdeng.cn/raf/model.html
    
    Expected directory structure:
        rafdb/
            train/
                1.angry/
                2.disgust/
                ...
            test/
                1.angry/
                ...
    """
    
    EMOTION_LABELS = {
        '1.angry': 0,
        '2.disgust': 1,
        '3.fear': 2,
        '4.happy': 3,
        '5.sad': 4,
        '6.surprise': 5,
        '7.neutral': 6,
    }
    
    def __init__(
        self,
        root: str = "./data/rafdb",
        split: str = "train",
        image_size: int = 224,
        augment: bool = True,
    ):
        self.root = Path(root) / split
        self.split = split
        self.image_size = image_size
        self.samples = []
        
        if not self.root.exists():
            print(f"Warning: RAFDB dataset not found at {self.root}")
            print("Please download from: http://www.whdeng.cn/raf/model.html")
            self.samples = []
        else:
            self._load_samples(augment)
    
    def _load_samples(self, augment: bool):
        for emotion_dir in self.root.iterdir():
            if emotion_dir.is_dir() and emotion_dir.name in self.EMOTION_LABELS:
                label = self.EMOTION_LABELS[emotion_dir.name]
                for img_path in emotion_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), label))
                for img_path in emotion_dir.glob("*.png"):
                    self.samples.append((str(img_path), label))
        
        if augment and self.split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if len(self.samples) == 0:
            img = torch.zeros(3, self.image_size, self.image_size)
            return {'image': img, 'label': 0}
        
        img_path, label = self.samples[idx]
        
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return {'image': image, 'label': label}


class RAFDBFewShot:
    """Few-shot RAFDB for limited data experiments."""
    
    def __init__(
        self,
        root: str = "./data/rafdb",
        n_way: int = 7,
        n_shot: int = 5,
        image_size: int = 224,
        augment: bool = False,
    ):
        from collections import defaultdict
        
        self.root = Path(root)
        self.n_way = n_way
        self.n_shot = n_shot
        self.image_size = image_size
        
        full_dataset = RAFDBDataset(
            root=root,
            split="train",
            image_size=image_size,
            augment=False,
        )
        
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(full_dataset.samples):
            class_indices[label].append(idx)
        
        selected_indices = []
        n_way = min(n_way, len(class_indices))
        for c in range(n_way):
            if c in class_indices:
                selected_indices.extend(class_indices[c][:n_shot])
        
        self.indices = selected_indices
        self.full_dataset = full_dataset
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.full_dataset[real_idx]


def get_rafdb_dataloaders(
    root: str = "./data/rafdb",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """Get RAFDB train and test dataloaders."""
    train_dataset = RAFDBDataset(
        root=root,
        split="train",
        image_size=image_size,
        augment=True,
    )
    
    test_dataset = RAFDBDataset(
        root=root,
        split="test",
        image_size=image_size,
        augment=False,
    )
    
    train_loader = None
    if len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
    
    test_loader = None
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    
    return train_loader, test_loader


def download_rafdb_instructions():
    """Print instructions for downloading RAFDB."""
    print("""
    RAFDB Dataset Download Instructions:
    ======================================
    
    1. Visit: http://www.whdeng.cn/raf/model.html
    2. Register for an account (required for download)
    3. Download the dataset (Basic Emotion.tar)
    4. Extract the archive
    5. Place the extracted folder at: ./data/rafdb/
    
    Expected structure:
        data/rafdb/
            train/
                1.angry/
                    train_001.jpg
                    ...
                2.disgust/
                ...
            test/
                1.angry/
                ...
    """)
