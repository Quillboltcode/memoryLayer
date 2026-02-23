# Model Architecture

Memory-VLM extends CLIP with learnable memory tokens that store contextual information directly in model weights.

## Overview

```
┌─────────────────┐     ┌─────────────────┐
│   Image Input   │     │   Text Input    │
│   (H x W x 3)   │     │   (seq_len)     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Patch Embed    │     │  Token Embed    │
│  (N x D)        │     │  (L x D)        │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  + [MEM] Tokens │     │  + [EOS] Token  │
│  (N + M x D)    │     │  (L + 1 x D)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Vision Encoder  │     │  Text Encoder   │
│ (Transformer)  │     │ (Transformer)   │
│                 │     │                 │
│ ┌───────────┐  │     │ ┌───────────┐   │
│ │  Memory   │  │     │ │           │   │
│ │ Attention │  │     │ │           │   │
│ └───────────┘  │     │ └───────────┘   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Pooled Image   │     │  Pooled Text    │
│    Embed (D)    │     │    Embed (D)     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │ Contrastive Loss     │
         │ (InfoNCE)             │
         └───────────────────────┘
```

## Components

### 1. Vision Encoder (MemoryCLIPVisionEncoder)

Built using `timm` Vision Transformer with memory token integration:

- **Patch Embedding**: Splits image into patches and projects to embedding dimension
- **Learnable Memory Tokens**: M learnable vectors (default: 64) that store contextual info
- **Transformer Blocks**: Standard ViT blocks with optional memory cross-attention
- **Pooling**: Attention pooling or [CLS] token pooling

```python
# Memory integration in forward pass
x = patch_embed(x)                    # (B, N, D)
memory_tokens = self.memory_tokens.expand(B, -1, -1)  # (B, M, D)
x = torch.cat([memory_tokens, x], dim=1)  # (B, N+M, D)
x = x + self.pos_embed
x = self.blocks(x)
image_embed = self.norm(x)
```

### 2. Text Encoder (MemoryCLIPTextEncoder)

Transformer encoder for text:

- **Token Embedding**: Learned embedding for vocabulary
- **Position Embedding**: Sinusoidal or learned positional embeddings
- **Transformer Blocks**: Standard transformer encoder layers
- **Pooling**: [EOS] token pooling for sentence-level embedding

### 3. Memory Module

The core innovation - learnable memory tokens that:

1. **Initialize**: Random initialization or pretrained
2. **Attend**: Cross-attend to input features
3. **Blend**: Combine with original features via learned gating
4. **Update**: Gradient-based learning of memory representations

```python
# Memory attention mechanism
memory, features = x[:, :M], x[:, M:]
attended_memory = self.memory_attn(memory, features, features)
memory = self.mem_alpha * attended_memory + (1 - self.mem_alpha) * memory
x = torch.cat([memory, features], dim=1)
```

### 4. Contrastive Loss

CLIP-style symmetric contrastive loss:

```python
def clip_loss(image_embeds, text_embeds, temperature=0.07):
    # Image-to-text loss
    logits = image_embeds @ text_embeds.T / temperature
    labels = torch.arange(len(image_embeds), device=image_embeds.device)
    loss_i2t = F.cross_entropy(logits, labels)
    
    # Text-to-image loss  
    logits = text_embeds @ image_embeds.T / temperature
    loss_t2i = F.cross_entropy(logits, labels)
    
    return (loss_i2t + loss_t2i) / 2
```

## Model Variants

| Model | Image Size | Embed Dim | Depth | Heads | Memory Tokens |
|-------|------------|-----------|-------|-------|---------------|
| ViT-B/16 | 224 | 512 | 12 | 8 | 64 |
| ViT-L/14 | 224 | 768 | 24 | 12 | 128 |
| ViT-H/14 | 224 | 1024 | 32 | 16 | 256 |

## Memory Update Strategies

### Attention-based (default)
```python
# Cross-attention from memory to features
attended = attention(query=memory, key=features, value=features)
memory = alpha * attended + (1 - alpha) * memory
```

### Gating-based
```python
# Learnable gate to control memory influence
gate = sigmoid(memory_gate @ memory)
memory = gate * attended + (1 - gate) * memory
```

### Persistent Memory
```python
# Memory tokens are updated only via gradient, no feature interaction
# Good for storing class prototypes or anchors
```

## Training Tips

1. **Memory Warmup**: Start with fewer memory tokens and increase
2. **Regularization**: Apply dropout to memory tokens to prevent overfitting
3. **Initialization**: Pretrain memory tokens on large datasets first
4. **Learning Rate**: Use lower LR for memory tokens initially

## References

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [ViT: An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
