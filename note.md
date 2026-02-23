## 1. The Titans Approach: Neural Memory

The "Titans" architecture (recently popularized by Google Research) replaces or augments the standard Transformer with a **Neural Memory** module. Unlike a KV cache that grows linearly, this memory uses "fast weights" to store information.

### How to adapt your VLM:

To integrate this into a VLM (like LLaVA or a generic CLIP+LLM setup), you replace the standard attention mechanism in the LLM backbone with a dual-memory system:

* **Short-term Memory:** A standard sliding-window attention to handle immediate visual cues and local text context.
* **Long-term (Neural) Memory:** A sub-network that "learns" to store past visual tokens and text. This memory is updated using a gradient-based rule:




Nested learning (often associated with hierarchical processing) focuses on "nesting" local details within global contexts. In a VLM, this translates to storing visual information at multiple levels of granularity.

### How to adapt your VLM:

1. **Level 1 (The "Inner" Loop):** High-resolution patches or frame-level details are processed and then "summarized" into memory.
2. **Level 2 (The "Outer" Loop):** The summarized embeddings are stored in a persistent buffer. When the model encounters a new image, it retrieves the most relevant "summaries" from the buffer.

This is particularly effective for **Long-Context VLMs** where you don't want to lose the specific details of an object seen 10 minutes ago in a video.

---

## 3. Practical Architecture: The Vision-Memory Bridge

To actually build this, you need to modify the "Projector" (the layer connecting the Vision Encoder to the LLM).

* **The Memory Buffer:** Create a fixed-size learnable tensor or a vector database (like Faiss)
* **Memory Augmentation:**
* **Input:** Current image tokens () + current text ().
* **Retrieval:** Use a cross-attention mechanism where the current tokens act as a "Query" to pull relevant history from the Memory Buffer.
* **Update:** Use a gating mechanism (like a GRU or the Titans update rule) to decide what parts of the current visual input are "important" enough to be written into the memory.

To achieve this, we will build a simplified **Titans-style Neural Memory** module. Instead of a traditional KV cache, this module uses "fast weights"—a hidden weight matrix that updates dynamically as it processes sequences of image patches or video frames.

Here is how the math translates before we look at the code. For a given input , the memory first retrieves information, and then updates its internal weights  using a simplified associative rule (like a linear attention mechanism):


Here is the PyTorch pseudo-code designed for a lightweight VLM.

### PyTorch Implementation: Small Memory-Augmented VLM

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TitansNeuralMemory(nn.Module):
    def __init__(self, embed_dim, memory_decay=0.9):
        super().__init__()
        self.embed_dim = embed_dim
        # \lambda controls how much past memory we retain
        self.memory_decay = memory_decay 
        
        # Learnable projections for the memory update
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Gate to decide how much new information to write
        self.write_gate = nn.Linear(embed_dim, 1)

    def forward(self, x, memory_state=None):
        """
        x: [batch_size, seq_len, embed_dim] (e.g., flattened image patches)
        memory_state: [batch_size, embed_dim, embed_dim] (the 'fast weights')
        """
        batch_size = x.size(0)
        
        # Initialize memory state if it's the first step
        if memory_state is None:
            memory_state = torch.zeros(batch_size, self.embed_dim, self.embed_dim, device=x.device)
            
        q = self.query_proj(x) # [B, Seq, Dim]
        k = self.key_proj(x)   # [B, Seq, Dim]
        v = self.value_proj(x) # [B, Seq, Dim]
        
        # 1. RETRIEVE from memory using current queries
        # memory_state acts as the weights we are multiplying against
        retrieved_info = torch.bmm(q, memory_state) 
        
        # Combine input with retrieved memory (simplified residual connection)
        out = x + retrieved_info
        
        # 2. UPDATE the memory state (The "Titans" fast-weight concept)
        # Calculate how strongly we want to write these specific tokens
        g = torch.sigmoid(self.write_gate(x)) # [B, Seq, 1]
        
        # Create the memory update (Outer product of K and V, scaled by gate)
        # Using einsum for cleaner batch outer product over sequence
        update = torch.einsum('bsi,bsj->bij', k * g, v) 
        
        # Apply decay to old memory and add the new update
        new_memory_state = (self.memory_decay * memory_state) + update
        
        return out, new_memory_state

class TinyMemoryVLM(nn.Module):
    def __init__(self, vocab_size=32000):
        super().__init__()
        # 1. Vision Encoder: A tiny ViT (e.g., ~50M params)
        self.embed_dim = 512
        self.vision_encoder = nn.Linear(768, self.embed_dim) # Stub for ViT output projector
        
        # 2. The Neural Memory Module
        self.memory_layer = TitansNeuralMemory(embed_dim=self.embed_dim)
        
        # 3. Small LLM Backbone (e.g., ~300M params, like a heavily truncated Llama/Qwen)
        self.text_embeddings = nn.Embedding(vocab_size, self.embed_dim)
        
        # A simple transformer block to represent the LLM processing
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=8, batch_first=True)
        self.llm_backbone = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.lm_head = nn.Linear(self.embed_dim, vocab_size)

    def forward(self, image_features, text_tokens, past_memory_state=None):
        # image_features shape: [B, num_patches, 768]
        # text_tokens shape: [B, seq_len]
        
        # 1. Project vision features to LLM dimension
        v_embeds = self.vision_encoder(image_features)
        
        # 2. Pass vision features through the Neural Memory
        # This compresses visual history and retrieves relevant past context
        v_mem_embeds, next_memory_state = self.memory_layer(v_embeds, past_memory_state)
        
        # 3. Embed text
        t_embeds = self.text_embeddings(text_tokens)
        
        # 4. Concatenate Memory-Augmented Vision + Text
        # Note: In a real autoregressive setup, you'd handle causal masking carefully here
        multimodal_context = torch.cat([v_mem_embeds, t_embeds], dim=1)
        
        # 5. Process through the LLM
        hidden_states = self.llm_backbone(multimodal_context)
        
        # 6. Predict next tokens
        logits = self.lm_head(hidden_states)
        
        return logits, next_memory_state

```

---

### How this keeps the parameter count low

1. **No KV Cache Bloat:** Standard transformers scale linearly (or worse) with sequence length because they store every Key and Value token. This implementation compresses the sequence into a fixed-size `embed_dim x embed_dim` matrix (the `memory_state`), costing exactly **0 extra parameters** during inference as the context grows.
2. **Modular Memory:** The `TitansNeuralMemory` layer only requires three small linear projections (`query`, `key`, `value`) and a tiny gating layer. For a 512-dimension model, this adds less than 1 million parameters.
3. **Backbone Sizing:** You can pair this memory layer with a 50M parameter vision encoder (like MobileViT) and a 300M parameter LLM (like Qwen2-0.5B), keeping your total footprint well under your 500M target.

To solve this, we use **Truncated Backpropagation Through Time (TBPTT)**.

The core idea is to process a long sequence in smaller "chunks." We pass the memory state forward continuously, but we **cut the gradient graph** (using `.detach()`) between chunks. This allows the model to learn long-term dependencies through the forward-flowing memory state without having to store the entire computational graph in VRAM.

Here is how you implement this training loop in PyTorch for your sub-500M VLM.

### PyTorch TBPTT Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assuming TinyMemoryVLM is defined as in the previous step
# model = TinyMemoryVLM(vocab_size=32000).cuda()
# optimizer = optim.AdamW(model.parameters(), lr=1e-4)
# criterion = nn.CrossEntropyLoss()

def train_tbptt(model, dataloader, optimizer, criterion, chunk_size, device="cuda"):
    model.train()
    
    for batch_idx, (video_features, text_tokens, target_tokens) in enumerate(dataloader):
        # video_features: [Batch, Total_Frames, Feature_Dim] 
        # text_tokens: [Batch, Total_Seq_Len]
        # target_tokens: [Batch, Total_Seq_Len]
        
        video_features = video_features.to(device)
        text_tokens = text_tokens.to(device)
        target_tokens = target_tokens.to(device)
        
        batch_size, total_frames, _ = video_features.size()
        
        # Initialize memory state as None for the very first chunk of the video
        memory_state = None 
        
        # Process the long sequence in manageable chunks
        for i in range(0, total_frames, chunk_size):
            # 1. Slice the current chunk
            end_idx = min(i + chunk_size, total_frames)
            chunk_video = video_features[:, i:end_idx, :]
            
            # (Assuming text is aligned with video chunks for simplicity)
            chunk_text = text_tokens[:, i:end_idx] 
            chunk_target = target_tokens[:, i:end_idx]
            
            # 2. DETACH the memory state (The magic of TBPTT)
            # This cuts the computational graph from the previous chunk.
            # Without this, memory usage would grow infinitely.
            if memory_state is not None:
                memory_state = memory_state.detach()
            
            # 3. Forward Pass: Pass in the memory state from the previous chunk
            optimizer.zero_grad()
            logits, memory_state = model(chunk_video, chunk_text, memory_state)
            
            # 4. Compute Loss
            # Flatten to compute CrossEntropy: [B * Seq_Len, Vocab] vs [B * Seq_Len]
            loss = criterion(logits.view(-1, logits.size(-1)), chunk_target.reshape(-1))
            
            # 5. Backward Pass
            loss.backward()
            
            # 6. Gradient Clipping (Crucial for Recurrent Memory)
            # Prevents exploding gradients when updating fast weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 7. Optimizer Step
            optimizer.step()
            
        print(f"Batch {batch_idx+1} Complete. Final Loss: {loss.item():.4f}")

```

---

### The Three Pillars of Training Recurrent Memory

* **The `.detach()` command:** When `memory_state` comes out of chunk 1, it carries the entire gradient history of chunk 1. Before passing it into chunk 2, we must call `.detach()`. This tells PyTorch to treat `memory_state` as a constant starting point for chunk 2, rather than a variable that requires gradients all the way back to the beginning of time.
* **Gradient Clipping:** Neural memory models are prone to exploding gradients because you are repeatedly multiplying matrices. `torch.nn.utils.clip_grad_norm_` acts as a safety valve, capping the maximum size of your gradient updates.
* **Chunk Alignment:** In a VLM, the hardest part of TBPTT is aligning the continuous visual memory (e.g., streaming video frames) with the discrete text queries. Your data loader needs to serve aligned chunks where the text asks about what is happening in the current, or recently passed, visual chunk.

