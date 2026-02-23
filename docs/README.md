# Memory-VLM

A Vision-Language Model (VLM) inspired by CLIP with in-weight memory augmentation. This project implements a memory-augmented dual encoder (image + text) that learns to store contextual information directly in model weights.

## Features

- **Memory-Augmented Vision Encoder**: Learnable memory tokens that integrate with vision features via attention
- **CLIP-style Dual Encoder**: Separate image and text encoders with contrastive learning
- **PyTorch + timm**: Built on proven foundations with pretrained ViT backbones
- **Einops**: Elegant tensor operations for readable code
- **Hydra**: Flexible configuration management

## Installation

```bash
# Create and activate uv environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

## Quick Start

```python
from models import MemoryCLIP

model = MemoryCLIP(
    embed_dim=512,
    image_size=224,
    vision_depth=12,
    text_depth=12,
    num_memory_tokens=64,
)

# Forward pass
import torch
images = torch.randn(2, 3, 224, 224)
texts = ["a cat", "a dog"]
image_embeds, text_embeds = model(images, texts)
```

## Training

```bash
# Single GPU
python train.py training.batch_size=128 training.epochs=100

# Multi-GPU
torchrun --nproc_per_node=4 train.py training.batch_size=512
```

## Kaggle Notebook Usage

### Method 1: Using sys.path.append (Recommended)

Sync the code to Kaggle and import directly:

```python
import sys
sys.path.append('/path/to/memoryvlm')

from models import MemoryCLIP
from train import train_epoch, get_config
import torch

# Create model
model = MemoryCLIP(embed_dim=512, num_memory_tokens=64)

# Simple training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Your data loader here...
for epoch in range(10):
    for batch in dataloader:
        images = batch['image'].to(device)
        texts = batch['text']
        loss = model.contrastive_loss(images, texts)
        loss.backward()
```

### Method 2: Run as Script with Custom Config

Create a custom config file and pass it to train.py:

```python
# In a Kaggle cell
!python train.py --config-path config_simple.yaml --epochs 50 --device cuda
```

### Method 3: Environment Variables

Override specific parameters via environment variables:

```python
import os
os.environ['MEMORY_VLM_EPOCHS'] = '50'
os.environ['MEMORY_VLM_BATCH_SIZE'] = '128'
os.environ['MEMORY_VLM_LR'] = '0.001'

# Then run
!python train.py --config-path config_simple.yaml
```

### Available Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_VLM_EPOCHS` | Number of training epochs | 100 |
| `MEMORY_VLM_BATCH_SIZE` | Batch size | 256 |
| `MEMORY_VLM_LR` | Learning rate | 5e-4 |
| `MEMORY_VLM_WEIGHT_DECAY` | Weight decay | 0.01 |
| `MEMORY_VLM_EMBED_DIM` | Model embedding dimension | 512 |
| `MEMORY_VLM_MEMORY_TOKENS` | Number of memory tokens | 64 |
| `MEMORY_VLM_IMAGE_SIZE` | Input image size | 224 |
| `MEMORY_VLM_PROJECT` | WandB project name | memory-vlm |
| `CONFIG_FILE` | Default config file path | config_simple.yaml |

### Testing Locally (CPU) vs Training on GPU

For quick testing and development:

```bash
# CPU training (slow, for debugging only)
python train.py --config-path config_simple.yaml --device cpu --epochs 1
```

For actual training:

```bash
# GPU training
python train.py --config-path config_simple.yaml --device cuda --epochs 100
```

### Custom Config File Example

Create `my_config.yaml`:

```yaml
model:
  name: "memory_clip_vit_b_16"
  embed_dim: 512
  num_memory_tokens: 64
  image_size: 224

training:
  epochs: 50
  batch_size: 128
  learning_rate: 0.0003
  weight_decay: 0.01

data:
  train_path: "data/train"
  val_path: "data/val"

logging:
  project: "my-project"
  name: "experiment-1"
  wandb: true
```

Then run:

```bash
python train.py --config-path my_config.yaml
```

## Configuration

See [CONFIG.md](CONFIG.md) for detailed configuration options.

## Architecture

See [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) for architectural details.

## Memory Mechanism

The memory tokens are learnable vectors that:
1. Are prepended to vision tokens before transformer layers
2. Attend to image patches via cross-attention
3. Allow the model to store and retrieve contextual information
4. Are updated during training through gradient-based learning

This enables the model to learn "in-weight" memories without external memory banks.
