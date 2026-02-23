# Configuration Guide

This document describes all configuration options for Memory-VLM.

## Config Files Structure

```
configs/
├── default.yaml      # Main config (Hydra defaults)
├── model.yaml        # Model architecture
├── training.yaml      # Training hyperparameters
└── project.yaml       # Project metadata
```

## Model Configuration

### Vision Encoder (`model`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | `"memory_clip_vit_b_16"` | Model identifier |
| `image_size` | int | `224` | Input image resolution |
| `patch_size` | int | `16` | ViT patch size |
| `embed_dim` | int | `512` | Embedding dimension |
| `depth` | int | `12` | Number of transformer layers |
| `num_heads` | int | `8` | Attention heads |
| `mlp_ratio` | float | `4.0` | MLP hidden dim multiplier |
| `drop_rate` | float | `0.0` | Dropout rate |
| `attn_drop_rate` | float | `0.0` | Attention dropout rate |

### Memory Configuration (`model.memory`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `true` | Enable memory tokens |
| `dim` | int | `512` | Memory token dimension |
| `num_memory_tokens` | int | `64` | Number of memory tokens |
| `memory_init` | str | `"learnable"` | Initialization strategy |
| `memory_update` | str | `"attention"` | How memory interacts with features |
| `memory_position` | str | `"prepend"` | Where memory tokens are placed |
| `mem_alpha` | float | `0.5` | Memory blending coefficient |
| `mem_reinit_steps` | int | `100` | Steps before memory hard reset |

### Text Encoder (`model.text`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | `49408` | Vocabulary size |
| `max_len` | int | `77` | Maximum sequence length |
| `embed_dim` | int | `512` | Embedding dimension |
| `depth` | int | `12` | Number of transformer layers |
| `num_heads` | int | `8` | Attention heads |
| `mlp_ratio` | float | `4.0` | MLP hidden dim multiplier |

### Contrastive Loss (`model.contrastive`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | `0.07` | Logit scaling temperature |
| `loss_type` | str | `"clip"` | Loss function type |
| `gather_distributed` | bool | `false` | Gather embeddings across GPUs |

## Training Configuration

### Base Training (`training`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | `100` | Training epochs |
| `batch_size` | int | `256` | Batch size per device |
| `accumulate_grad_batches` | int | `1` | Gradient accumulation |
| `learning_rate` | float | `5e-4` | Learning rate |
| `weight_decay` | float | `0.01` | Weight decay |
| `clip_grad` | float | `1.0` | Gradient clipping norm |

### Optimizer (`training.optimizer`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | str | `"adamw"` | Optimizer name |
| `optimizer_kwargs.betas` | list | `[0.9, 0.999]` | Adam betas |
| `optimizer_kwargs.eps` | float | `1e-8` | Adam epsilon |

### Scheduler (`training.scheduler`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scheduler` | str | `"cosine"` | Scheduler name |
| `scheduler_kwargs.warmup_epochs` | int | `5` | Warmup epochs |
| `scheduler_kwargs.min_lr` | float | `1e-6` | Minimum learning rate |

### Data (`training.data`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_path` | str | `"data/train"` | Training data path |
| `val_path` | str | `"data/val"` | Validation data path |
| `image_size` | int | `224` | Input image size |
| `augmentation` | str | `"clip"` | Augmentation strategy |
| `num_workers` | int | `4` | DataLoader workers |
| `prefetch_factor` | int | `2` | Prefetch batches |
| `pin_memory` | bool | `true` | Pin memory for faster transfer |
| `persistent_workers` | bool | `true` | Keep workers alive |

### Logging (`training.logging`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_interval` | int | `10` | Logging frequency (steps) |
| `eval_interval` | int | `1000` | Evaluation frequency |
| `save_interval` | int | `1000` | Checkpoint frequency |
| `project` | str | `"memory-vlm"` | W&B/TB project name |
| `entity` | str | `null` | W&B entity |
| `name` | str | `"memory_clip"` | Experiment name |
| `wandb` | bool | `true` | Use Weights & Biases |
| `tensorboard` | bool | `true` | Use TensorBoard |

### Checkpointing (`training.checkpoint`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_top_k` | int | `3` | Number of best checkpoints to save |
| `monitor` | str | `"val_loss"` | Metric to monitor |
| `mode` | str | `"min"` | Monitoring mode |
| `save_last` | bool | `true` | Always save last checkpoint |
| `dirpath` | str | `"checkpoints"` | Checkpoint directory |
| `filename` | str | `...` | Checkpoint filename pattern |

## Overriding Configs

```bash
# Override single values
python train.py training.epochs=50 model.memory.num_memory_tokens=128

# Override nested values
python train.py training.optimizer_kwargs.betas="[0.9, 0.995]"

# Use config groups
python train.py --config-name=training model=large

# Debug mode
python train.py training.batch_size=2 training.epochs=1
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `WANDB_API_KEY` | Weights & Biases API key |
| `TRANSFORMERS_CACHE` | HuggingFace cache directory |
