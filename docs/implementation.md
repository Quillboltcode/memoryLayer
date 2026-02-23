# Memory-Augmented ViT Implementation

## Overview

This project implements **Hopfield-based associative memory** into Vision Transformers (ViT) for limited data learning and continual learning scenarios.

## Architecture

### Conceptual Framework

| Component | Function |
|-----------|----------|
| **Self-Attention** | Working memory (processes current input) |
| **Hopfield Layer** | Long-term memory (retrieves stored patterns) |
| **FFN** | Local computation |

## Models

### 1. MemoryAugmentedViT (Primary Model)

Vision Transformer with Hopfield-based associative memory inserted after attention or FFN.

```python
from models import MemoryAugmentedViT

model = MemoryAugmentedViT(
    backbone_name='vit_base_patch16_224',
    pretrained=True,
    num_classes=10,
    mem_size=1000,
    hopfield_position='after_attn',  # or 'after_ffn'
    beta=0.125,
    freeze_backbone=True,
)
```

**Key Features:**
- Inserts Hopfield layers into timm ViT backbone
- Supports two memory modes: `associative` (Hebbian updates) and `class_prototype` (class centroids)
- Configurable memory size and temperature (beta)

### 2. Baseline Models

#### LinearProbeViT
Frozen backbone with trainable linear classification head.

```python
from models.baselines import LinearProbeViT

model = LinearProbeViT(
    backbone_name='vit_base_patch16_224',
    pretrained=True,
    num_classes=10,
)
```

**Trainable Parameters:** Only `head.weight` and `head.bias`

#### VPT-Shallow
Visual Prompt Tuning at the first layer only.

```python
from models.baselines import VPTShallow

model = VPTShallow(
    backbone_name='vit_base_patch16_224',
    pretrained=True,
    num_classes=10,
    num_prompts=10,
)
```

**Trainable Parameters:** `prompts` + `head` weights

#### VPT-Deep
Visual Prompt Tuning at every transformer block.

```python
from models.baselines import VPTDeep

model = VPTDeep(
    backbone_name='vit_base_patch16_224',
    pretrained=True,
    num_classes=10,
    num_prompts=10,
)
```

### 3. Continual Learning Baselines

```python
from models.baselines.continual import EWC, LwF, iCaRL, ReplayBuffer

# Elastic Weight Consolidation
ewc = EWC(model, dataloader, ewc_lambda=1000)
loss = ewc.penalty()

# Learning without Forgetting
lwf = LwF(model, old_model, alpha=1.0, temperature=2.0)

# iCaRL
icarl = iCaRL(model, num_classes=100, memory_size=2000)
```

## Training

### Quick Start

```bash
# Train Memory-Augmented ViT on CIFAR-10
python train_memory_vit.py --config-path config_memory_vit.yaml

# Train Linear Probe baseline
python train_memory_vit.py --model linear_probe --dataset cifar10 --epochs 50

# Train VPT baseline
python train_memory_vit.py --model vpt_shallow --dataset cifar100

# Few-shot learning
python train_memory_vit.py --model memory_vit --dataset cifar10 --few-shot --n-shot 5
```

### Configuration

Edit `config_memory_vit.yaml`:

```yaml
model:
  name: "memory_vit"          # memory_vit, linear_probe, vpt_shallow, vpt_deep
  backbone: "vit_base_patch16_224"
  num_classes: 10
  mem_size: 1000              # Hopfield memory size
  hopfield_position: "after_attn"
  beta: 0.125
  freeze_backbone: true

training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001

data:
  dataset: "cifar10"          # cifar10, cifar100, rafdb
  image_size: 224
```

## Datasets

### CIFAR-10/100
- Automatic download via torchvision
- Resized to 224x224 for ViT
- Data augmentation enabled for training

### RAFDB (Facial Expression)
- Requires manual download from http://www.whdeng.cn/raf/model.html
- 7 basic emotions: angry, disgust, fear, happy, sad, surprise, neutral

### Few-Shot Learning
```python
from data.cifar import FewShotCIFAR10, FewShotCIFAR100

dataset = FewShotCIFAR10(
    root='./data',
    n_way=10,
    n_shot=5,
    image_size=224,
)
```

## Key Hyperparameters

| Parameter | CIFAR-10 | CIFAR-100 | Description |
|-----------|----------|-----------|-------------|
| Image Size | 224 | 224 | ViT input size |
| Batch Size | 64 | 64 | Training batch |
| Learning Rate | 1e-3 | 1e-3 | AdamW lr |
| Memory Size | 1000 | 1000 | Hopfield patterns |
| Beta | 0.125 | 0.125 | Temperature |
| Dropout | 0.1 | 0.1 | Regularization |

## Expected Results

| Method | CIFAR-10 (5-shot) | CIFAR-100 (10-shot) |
|--------|-------------------|---------------------|
| Frozen + Linear | ~85% | ~65% |
| VPT-Deep | ~92% | ~78% |
| **Frozen + Hopfield** | **~90%** | **~75%** |

## File Structure

```
memoryvlm/
├── models/
│   ├── HopfieldLayer.py         # Hopfield associative memory
│   ├── vit_blocks.py            # Memory-augmented block
│   ├── memory_vit.py            # Full model
│   └── baselines/
│       ├── linear_probe.py      # Linear probe baseline
│       ├── vpt.py               # VPT baselines
│       └── continual/           # CL methods
├── data/
│   ├── cifar.py                # CIFAR datasets
│   └── rafdb.py                # RAFDB dataset
├── training/
│   └── __init__.py             # Trainer classes
├── train_memory_vit.py          # Main training script
└── config_memory_vit.yaml       # Configuration
```
