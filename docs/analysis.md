# 🔍 Comprehensive Weight Analysis for Hopfield-Augmented Vision Transformers

Analyzing weights in a Hopfield-augmented ViT requires specialized techniques that go beyond standard CNN/Transformer analysis. Below is a **complete, actionable framework** for understanding your model's behavior, with code examples you can implement immediately.

---

## 🧠 Why Standard Weight Analysis Fails for Hopfield Networks

Standard weight analysis techniques (like filter visualization in CNNs) don't apply directly because:
- **Memory patterns aren't filters** - they're abstract associative patterns
- **Query projection is asymmetric** (no key/value projections)
- **Temperature parameter** controls retrieval dynamics
- **Memory utilization** varies per input (unlike fixed filters)

---

## ✅ 1. Memory Pattern Analysis (Most Critical)

### A. Memory Pattern Distribution
```python
def analyze_memory_distributions(model):
    """Analyze statistical properties of memory patterns across layers"""
    results = {}
    
    for layer_idx, block in enumerate(model.backbone.blocks):
        if not hasattr(block, 'hopfield_layer'):
            continue
            
        hopfield = block.hopfield_layer
        
        # Get memory statistics
        key_mean = hopfield.memory_keys.mean().item()
        key_std = hopfield.memory_keys.std().item()
        key_norm = hopfield.memory_keys.norm(dim=-1).mean().item()
        
        val_mean = hopfield.memory_vals.mean().item()
        val_std = hopfield.memory_vals.std().item()
        val_norm = hopfield.memory_vals.norm(dim=-1).mean().item()
        
        # Check if auto-associative (keys == values)
        is_auto = torch.allclose(hopfield.memory_keys, hopfield.memory_vals, atol=1e-3)
        
        results[f"layer_{layer_idx}"] = {
            "key_mean": key_mean,
            "key_std": key_std,
            "key_norm": key_norm,
            "val_mean": val_mean,
            "val_std": val_std,
            "val_norm": val_norm,
            "is_auto_associative": is_auto,
            "memory_size": hopfield.mem_size
        }
    
    return results

# Usage
memory_stats = analyze_memory_distributions(model)
for layer, stats in memory_stats.items():
    print(f"{layer}: key_norm={stats['key_norm']:.4f}, auto={stats['is_auto_associative']}")
```

### B. Memory Pattern Similarity Analysis
```python
def analyze_memory_similarity(model, save_path=None):
    """Analyze pairwise similarity between memory patterns"""
    plt.figure(figsize=(15, 10))
    
    for layer_idx, block in enumerate(model.backbone.blocks):
        if not hasattr(block, 'hopfield_layer'):
            continue
            
        hopfield = block.hopfield_layer
        mem_k = hopfield.memory_keys[0].detach()  # (mem_size, dim)
        
        # Compute cosine similarity matrix
        mem_k_norm = F.normalize(mem_k, p=2, dim=-1)
        sim_matrix = torch.mm(mem_k_norm, mem_k_norm.t())  # (mem_size, mem_size)
        
        # Plot similarity heatmap
        plt.subplot(3, 4, layer_idx+1)
        plt.imshow(sim_matrix.cpu().numpy(), cmap='viridis', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title(f'Layer {layer_idx} Memory Similarity')
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    # Quantify diversity: higher entropy = more diverse patterns
    entropy = -torch.mean(sim_matrix * torch.log2(torch.clamp(sim_matrix, 1e-7, 1)))
    print(f"Memory pattern entropy: {entropy.item():.4f} (higher = more diverse)")
```

### C. Memory Pattern Evolution Tracking
```python
def track_memory_evolution(model, checkpoint_paths, output_file="memory_evolution.npz"):
    """
    Track how memory patterns evolve during training
    checkpoint_paths: list of model checkpoint paths in training order
    """
    all_memories = []
    
    for ckpt_path in checkpoint_paths:
        # Load checkpoint
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict, strict=False)
        
        # Extract memory patterns from each layer
        layer_memories = []
        for block in model.backbone.blocks:
            if hasattr(block, 'hopfield_layer'):
                mem = block.hopfield_layer.memory_keys[0].detach().cpu()
                layer_memories.append(mem)
        
        all_memories.append(layer_memories)
    
    # Save for later analysis
    np.savez(output_file, memories=all_memories)
    
    # Analyze stability: cosine similarity between consecutive checkpoints
    stability_metrics = []
    for i in range(1, len(all_memories)):
        layer_stability = []
        for layer in range(len(all_memories[0])):
            prev = F.normalize(all_memories[i-1][layer], p=2, dim=-1)
            curr = F.normalize(all_memories[i][layer], p=2, dim=-1)
            sim = torch.diag(torch.mm(prev, curr.t())).mean().item()
            layer_stability.append(sim)
        stability_metrics.append(layer_stability)
    
    # Plot stability across training
    plt.figure(figsize=(10, 6))
    plt.plot(stability_metrics)
    plt.xlabel('Training Step')
    plt.ylabel('Memory Pattern Stability (cosine sim)')
    plt.title('Memory Pattern Evolution During Training')
    plt.savefig('memory_stability.png')
    plt.show()
```

---

## ✅ 2. Query Projection Analysis

### A. Feature Importance via Query Weights
```python
def analyze_query_weights(model, class_names, dataset, num_samples=100):
    """Identify which input features most influence memory retrieval"""
    # Get random samples
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4
    )
    
    all_grads = []
    for i, (images, _) in enumerate(loader):
        if i * 32 >= num_samples:
            break
            
        images = images.to(device)
        images.requires_grad = True
        
        # Forward through modified model that returns attention
        with torch.enable_grad():
            features = model.backbone.patch_embed(images)
            features = model.backbone._pos_embed(features)
            
            # Analyze first Hopfield layer
            block = model.backbone.blocks[0]
            hopfield = block.hopfield_layer
            
            # Get query projections
            q = hopfield.to_q(block.vit_block.norm1(features))
            
            # Compute attention (for gradient analysis)
            attn = torch.matmul(q, hopfield.memory_keys.transpose(-2, -1)) * hopfield.temp
            attn = F.softmax(attn, dim=-1)
            
            # Backpropagate through attention to get input gradients
            loss = attn.sum()
            loss.backward()
            
            # Store gradients
            all_grads.append(images.grad.abs().mean(dim=(0, 2, 3)).cpu())
    
    # Average gradients across samples
    avg_grads = torch.stack(all_grads).mean(0)
    
    # Visualize most important channels
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(3), avg_grads[:3])
    plt.xticks(range(3), ['R', 'G', 'B'])
    plt.title('Channel Importance')
    
    # For ViT, analyze patch importance
    patch_importance = avg_grads[3:].view(14, 14)  # For 224px/16=14
    plt.subplot(1, 2, 2)
    plt.imshow(patch_importance, cmap='hot')
    plt.colorbar()
    plt.title('Spatial Importance Map')
    plt.savefig('query_importance.png')
    plt.show()
```

### B. Query Subspace Analysis
```python
def analyze_query_subspaces(model, dataset):
    """Analyze the subspace spanned by query projections"""
    from sklearn.decomposition import PCA
    import seaborn as sns
    
    # Collect query vectors
    all_queries = []
    for images, _ in torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4):
        images = images.to(device)
        features = model.backbone.patch_embed(images)
        features = model.backbone._pos_embed(features)
        
        # Get queries from first Hopfield layer
        block = model.backbone.blocks[0]
        q = block.hopfield_layer.to_q(block.vit_block.norm1(features))
        
        # Only take class token queries for simplicity
        all_queries.append(q[:, 0].detach().cpu())
        
        if len(all_queries) * 32 > 500:  # Sample size
            break
    
    queries = torch.cat(all_queries, dim=0)
    
    # Apply PCA
    pca = PCA(n_components=50)
    reduced = pca.fit_transform(queries.numpy())
    
    # Plot explained variance
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Query Vector Dimensionality')
    
    # Plot first 2 components colored by class (if labels available)
    plt.subplot(1, 2, 2)
    try:
        # Assuming you have access to labels
        _, labels = next(iter(torch.utils.data.DataLoader(
            dataset, batch_size=500, shuffle=True
        )))
        sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette='viridis')
    except:
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
    plt.title('Query Vectors in PCA Space')
    plt.savefig('query_subspaces.png')
    plt.show()
    
    # Report intrinsic dimensionality
    threshold = 0.9  # 90% variance explained
    dims_needed = np.where(np.cumsum(pca.explained_variance_ratio_) >= threshold)[0][0] + 1
    print(f"Query vectors live in {dims_needed}-dimensional subspace ({threshold*100}% variance)")
```

---

## ✅ 3. Temperature Parameter Analysis

### A. Temperature Dynamics Monitoring
```python
def monitor_temperature_dynamics(model, train_loader, optimizer):
    """Track how temperature affects retrieval during training"""
    temperature_history = []
    entropy_history = []
    
    model.eval()
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            
            for block in model.backbone.blocks:
                if hasattr(block, 'hopfield_layer'):
                    hopfield = block.hopfield_layer
                    
                    # Forward pass to get attention
                    features = model.backbone.patch_embed(images)
                    features = model.backbone._pos_embed(features)
                    x = block.vit_block.norm1(features)
                    q = hopfield.to_q(x)
                    
                    # Compute attention
                    attn = torch.matmul(q, hopfield.memory_keys.transpose(-2, -1)) * hopfield.temp
                    attn = F.softmax(attn, dim=-1)
                    
                    # Calculate entropy of attention distributions
                    entropy = -torch.sum(attn * torch.log(attn + 1e-7), dim=-1).mean().item()
                    
                    temperature_history.append(hopfield.temp.item())
                    entropy_history.append(entropy)
            
            break  # One batch is enough for analysis
    
    # Plot relationship
    plt.figure(figsize=(10, 5))
    plt.scatter(temperature_history, entropy_history, alpha=0.6)
    plt.xlabel('Temperature Parameter')
    plt.ylabel('Attention Entropy')
    plt.title('Temperature vs. Retrieval Entropy')
    plt.savefig('temperature_entropy.png')
    plt.show()
    
    # Calculate correlation
    corr = np.corrcoef(temperature_history, entropy_history)[0, 1]
    print(f"Temperature-entropy correlation: {corr:.4f} (should be positive)")
    
    return temperature_history, entropy_history
```

### B. Temperature Sensitivity Analysis
```python
def analyze_temperature_sensitivity(model, images, layer_idx=0):
    """Test how model predictions change with temperature"""
    original_temp = model.backbone.blocks[layer_idx].hopfield_layer.temp.item()
    temperatures = np.logspace(-2, 1, 20)  # From 0.01 to 10
    
    predictions = []
    for temp in temperatures:
        # Set temperature
        model.backbone.blocks[layer_idx].hopfield_layer.temp.data = torch.tensor([temp])
        
        # Get prediction
        with torch.no_grad():
            logits = model(images)
            probs = F.softmax(logits, dim=-1)
            predictions.append(probs.cpu())
    
    # Reset to original
    model.backbone.blocks[layer_idx].hopfield_layer.temp.data = torch.tensor([original_temp])
    
    # Analyze prediction stability
    predictions = torch.stack(predictions)
    stability = F.cosine_similarity(
        predictions[0].unsqueeze(0), 
        predictions, 
        dim=-1
    ).mean(dim=1)
    
    plt.figure(figsize=(10, 5))
    plt.semilogx(temperatures, stability)
    plt.xlabel('Temperature (log scale)')
    plt.ylabel('Prediction Stability')
    plt.title('Model Sensitivity to Temperature Parameter')
    plt.grid(True, which="both", ls="-")
    plt.savefig('temperature_sensitivity.png')
    plt.show()
    
    # Find optimal temperature range
    stable_idx = np.where(stability > 0.95)[0]
    if len(stable_idx) > 0:
        print(f"Stable temperature range: {temperatures[stable_idx[0]]:.4f} - {temperatures[stable_idx[-1]]:.4f}")
    else:
        print("No stable temperature range found - model is highly sensitive")
```

---

## ✅ 4. Memory Utilization Analysis (Critical for Hopfield Networks)

### A. Memory Slot Activation Tracking
```python
def track_memory_utilization(model, dataloader, num_batches=10):
    """Track which memory slots get activated across different inputs"""
    activation_counts = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            images = images.to(device)
            features = model.backbone.patch_embed(images)
            features = model.backbone._pos_embed(features)
            
            batch_activations = []
            for block_idx, block in enumerate(model.backbone.blocks):
                if not hasattr(block, 'hopfield_layer'):
                    continue
                    
                # Process through previous blocks
                x = features
                for prev_block in model.backbone.blocks[:block_idx]:
                    x = prev_block(x)
                
                # Get attention for this Hopfield layer
                x_norm = block.vit_block.norm1(x)
                q = block.hopfield_layer.to_q(x_norm)
                attn = torch.matmul(q, block.hopfield_layer.memory_keys.transpose(-2, -1))
                attn = F.softmax(attn * block.hopfield_layer.temp, dim=-1)
                
                # Track which memory slots are used (top-5 per token)
                _, top_indices = torch.topk(attn, k=5, dim=-1)
                activations = torch.zeros_like(block.hopfield_layer.memory_keys[0, :, 0])
                for idx in top_indices.view(-1):
                    activations[idx] += 1
                
                batch_activations.append(activations.cpu())
            
            activation_counts.append(batch_activations)
    
    # Aggregate across batches
    total_activations = []
    for layer_idx in range(len(activation_counts[0])):
        layer_acts = torch.stack([batch[layer_idx] for batch in activation_counts]).sum(0)
        total_activations.append(layer_acts)
    
    # Plot utilization histograms
    plt.figure(figsize=(15, 10))
    for layer_idx, acts in enumerate(total_activations):
        plt.subplot(3, 4, layer_idx+1)
        plt.hist(acts, bins=50)
        plt.title(f'Layer {layer_idx} Memory Utilization')
        plt.xlabel('Activation Count')
        plt.ylabel('Memory Slots')
    
    plt.tight_layout()
    plt.savefig('memory_utilization.png')
    plt.show()
    
    # Calculate utilization metrics
    for layer_idx, acts in enumerate(total_activations):
        utilization = (acts > 0).float().mean().item()
        entropy = -torch.sum(acts * torch.log(acts + 1e-7)) / acts.sum()
        print(f"Layer {layer_idx}: {utilization:.2%} slots used, entropy={entropy:.4f}")
```

### B. Memory Slot Specialization Analysis
```python
def analyze_memory_specialization(model, dataloader, class_names):
    """Determine if memory slots specialize for specific classes"""
    # First, collect which memory slots activate for which classes
    class_activations = {cls: [] for cls in range(len(class_names))}
    
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            features = model.backbone.patch_embed(images)
            features = model.backbone._pos_embed(features)
            
            # Process through all blocks
            for block in model.backbone.blocks:
                if hasattr(block, 'hopfield_layer'):
                    x = block.vit_block.norm1(features)
                    q = block.hopfield_layer.to_q(x)
                    attn = torch.matmul(q, block.hopfield_layer.memory_keys.transpose(-2, -1))
                    attn = F.softmax(attn * block.hopfield_layer.temp, dim=-1)
                    
                    # Record top memory slot for class token
                    for i in range(labels.shape[0]):
                        cls_idx = labels[i].item()
                        top_slot = torch.argmax(attn[i, 0]).item()
                        class_activations[cls_idx].append(top_slot)
            
            break  # One batch is enough for analysis
    
    # Analyze specialization
    specialization = []
    for cls_idx, activations in class_activations.items():
        if not activations:
            continue
            
        # Count how often each slot is used for this class
        slot_counts = torch.bincount(torch.tensor(activations), 
                                   minlength=model.mem_size)
        
        # Calculate specialization score: entropy relative to uniform
        probs = slot_counts.float() / slot_counts.sum()
        entropy = -torch.sum(probs * torch.log(probs + 1e-7))
        max_entropy = torch.log(torch.tensor(model.mem_size, dtype=torch.float))
        specialization_score = 1.0 - (entropy / max_entropy)
        
        specialization.append((cls_idx, specialization_score.item()))
    
    # Sort by specialization
    specialization.sort(key=lambda x: x[1], reverse=True)
    
    # Plot top specialized classes
    top_classes = specialization[:10]
    plt.figure(figsize=(12, 6))
    plt.bar(
        [class_names[cls] for cls, _ in top_classes],
        [score for _, score in top_classes]
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Memory Specialization by Class')
    plt.ylabel('Specialization Score (0-1)')
    plt.tight_layout()
    plt.savefig('memory_specialization.png')
    plt.show()
    
    print(f"Most specialized class: {class_names[specialization[0][0]]} "
          f"(score={specialization[0][1]:.4f})")
    print(f"Least specialized class: {class_names[specialization[-1][0]]} "
          f"(score={specialization[-1][1]:.4f})")
```

---

## ✅ 5. Cross-Component Analysis (Hopfield vs. Standard ViT)

### A. Attention Pattern Comparison
```python
def compare_attention_patterns(model, images, layer_idx=3):
    """Compare standard ViT attention with Hopfield retrieval patterns"""
    # Get standard ViT attention (need to modify model temporarily)
    original_block = model.backbone.blocks[layer_idx]
    
    # Store original Hopfield layer
    original_hopfield = original_block.hopfield_layer
    
    # Create a version without Hopfield for comparison
    class NoHopfieldBlock(nn.Module):
        def __init__(self, vit_block):
            super().__init__()
            self.vit_block = vit_block
            
        def forward(self, x):
            x = self.vit_block.norm1(x)
            x = self.vit_block.attn(x)
            x = x + self.vit_block.drop_path(self.vit_block.mlp(self.vit_block.norm2(x)))
            return x
    
    # Replace with no-Hopfield version
    model.backbone.blocks[layer_idx] = NoHopfieldBlock(original_block.vit_block)
    
    # Get standard attention
    with torch.no_grad():
        features = model.backbone.patch_embed(images)
        features = model.backbone._pos_embed(features)
        
        # Process through previous blocks
        for i in range(layer_idx):
            features = model.backbone.blocks[i](features)
        
        # Get attention from standard ViT
        x = original_block.vit_block.norm1(features)
        attn_vit = original_block.vit_block.attn.get_attention(x)
        
        # Get Hopfield attention (restore original block)
        model.backbone.blocks[layer_idx] = original_block
        x = original_block.vit_block.norm1(features)
        q = original_hopfield.to_q(x)
        attn_hopfield = F.softmax(
            torch.matmul(q, original_hopfield.memory_keys.transpose(-2, -1)) * 
            original_hopfield.temp, 
            dim=-1
        )
    
    # Compare attention for class token
    plt.figure(figsize=(15, 5))
    
    # Standard ViT attention
    plt.subplot(1, 3, 1)
    plt.imshow(attn_vit[0, 0].cpu().numpy())
    plt.colorbar()
    plt.title('Standard ViT Attention (Head 0)')
    
    # Hopfield attention
    plt.subplot(1, 3, 2)
    plt.imshow(attn_hopfield[0, 0].cpu().numpy().reshape(14, 14), cmap='hot')
    plt.colorbar()
    plt.title('Hopfield Memory Retrieval')
    
    # Difference
    hopfield_reshaped = attn_hopfield[0, 0].cpu().numpy().reshape(196)
    diff = attn_vit[0, 0].cpu().numpy() - hopfield_reshaped[:196]
    plt.subplot(1, 3, 3)
    plt.imshow(diff.reshape(14, 14), cmap='coolwarm', vmin=-0.1, vmax=0.1)
    plt.colorbar()
    plt.title('Attention Difference')
    
    plt.tight_layout()
    plt.savefig('attention_comparison.png')
    plt.show()
    
    # Quantify similarity
    cosine_sim = F.cosine_similarity(
        attn_vit[0, 0].flatten(), 
        torch.tensor(hopfield_reshaped), 
        dim=0
    ).item()
    print(f"Attention pattern similarity: {cosine_sim:.4f}")
```

### B. Feature Space Alignment
```python
def analyze_feature_alignment(model, dataloader):
    """Analyze how Hopfield layers transform the feature space"""
    from sklearn.neighbors import KNeighborsClassifier
    
    # Collect features with and without Hopfield
    features_no_hopfield = []
    features_with_hopfield = []
    labels_list = []
    
    # Temporarily remove Hopfield layers
    original_blocks = [block for block in model.backbone.blocks]
    
    class NoHopfieldBlock(nn.Module):
        def __init__(self, vit_block):
            super().__init__()
            self.vit_block = vit_block
            
        def forward(self, x):
            x = self.vit_block.norm1(x)
            x = self.vit_block.attn(x)
            x = x + self.vit_block.drop_path(self.vit_block.mlp(self.vit_block.norm2(x)))
            return x
    
    # Create no-Hopfield version
    no_hopfield_blocks = nn.ModuleList([
        NoHopfieldBlock(block.vit_block) if hasattr(block, 'hopfield_layer') else block
        for block in original_blocks
    ])
    
    model.backbone.blocks = no_hopfield_blocks
    
    # Collect no-Hopfield features
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Get features without Hopfield
            features = model.backbone.forward_features(images)
            if isinstance(features, tuple):
                features = features[0]
            cls_token = features[:, 0]
            features_no_hopfield.append(cls_token.cpu())
            labels_list.append(labels.cpu())
    
    # Restore Hopfield layers
    model.backbone.blocks = nn.ModuleList(original_blocks)
    
    # Collect with-Hopfield features
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            features = model.backbone.forward_features(images)
            if isinstance(features, tuple):
                features = features[0]
            cls_token = features[:, 0]
            features_with_hopfield.append(cls_token.cpu())
    
    # Convert to tensors
    features_no_hopfield = torch.cat(features_no_hopfield, dim=0)
    features_with_hopfield = torch.cat(features_with_hopfield, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    # Analyze with kNN
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Train kNN on no-Hopfield features
    knn.fit(features_no_hopfield.numpy(), labels.numpy())
    acc_no_hopfield = knn.score(features_no_hopfield.numpy(), labels.numpy())
    
    # Test how well no-Hopfield kNN works on Hopfield features
    acc_transfer = knn.score(features_with_hopfield.numpy(), labels.numpy())
    
    print(f"kNN accuracy (no Hopfield): {acc_no_hopfield:.4f}")
    print(f"kNN transfer accuracy (to Hopfield features): {acc_transfer:.4f}")
    print(f"Feature space preservation: {acc_transfer/acc_no_hopfield:.4f}")
    
    # Visualize with t-SNE
    from sklearn.manifold import TSNE
    combined = torch.cat([features_no_hopfield, features_with_hopfield], dim=0)
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(combined.numpy())
    
    plt.figure(figsize=(12, 6))
    plt.scatter(embedded[:len(features_no_hopfield), 0], 
                embedded[:len(features_no_hopfield), 1],
                c=labels, cmap='tab10', alpha=0.5, label='No Hopfield')
    plt.scatter(embedded[len(features_no_hopfield):, 0], 
                embedded[len(features_no_hopfield):, 1],
                c=labels, cmap='tab10', alpha=0.5, marker='x', label='With Hopfield')
    plt.legend()
    plt.title('Feature Space Alignment (t-SNE)')
    plt.savefig('feature_alignment.png')
    plt.show()
```

---

## ✅ 6. Advanced Diagnostic Tools

### A. Memory Pattern Visualization (for vision)
```python
def visualize_memory_patterns(model, dataset, num_patterns=16, layer_idx=6):
    """Visualize what memory patterns 'look like' in image space"""
    hopfield = model.backbone.blocks[layer_idx].hopfield_layer
    
    # Get random images to establish baseline
    rand_idx = torch.randperm(len(dataset))[:100]
    rand_images = torch.stack([dataset[i][0] for i in rand_idx]).to(device)
    
    # Get average feature representation
    with torch.no_grad():
        rand_features = model.backbone.patch_embed(rand_images)
        rand_features = model.backbone._pos_embed(rand_features)
        for block in model.backbone.blocks[:layer_idx]:
            rand_features = block(rand_features)
        rand_features = model.backbone.blocks[layer_idx].vit_block.norm1(rand_features)
    
    # Create query vectors that maximally activate each memory slot
    max_queries = []
    for slot_idx in range(hopfield.mem_size):
        # Create one-hot attention target
        target_attn = torch.zeros(hopfield.mem_size, device=device)
        target_attn[slot_idx] = 1.0
        
        # Solve for query that produces this attention
        # qK^T = target_attn => q = target_attn @ K (since K is normalized)
        k = F.normalize(hopfield.memory_keys[0], p=2, dim=-1)
        q = target_attn @ k  # (dim,)
        
        max_queries.append(q)
    
    max_queries = torch.stack(max_queries)  # (mem_size, dim)
    
    # Find real images whose queries are closest to these max queries
    best_matches = []
    for q in max_queries:
        # Project random features to query space
        rand_queries = hopfield.to_q(rand_features)
        
        # Find closest match
        sims = F.cosine_similarity(
            rand_queries.reshape(-1, rand_queries.shape[-1]), 
            q.unsqueeze(0),
            dim=-1
        )
        _, idx = torch.topk(sims, k=1)
        best_matches.append(rand_idx[idx // 197])  # Convert flat idx to image idx
    
    # Visualize top matches for selected patterns
    plt.figure(figsize=(15, 10))
    for i in range(min(num_patterns, hopfield.mem_size)):
        img, _ = dataset[best_matches[i]]
        plt.subplot(4, 4, i+1)
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(f'Memory {i}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('memory_visualization.png')
    plt.show()
```

### B. Memory Capacity Testing
```python
def test_memory_capacity(model, train_loader, test_loader, max_patterns=500):
    """Test how many patterns the memory can effectively store"""
    from sklearn.linear_model import LogisticRegression
    
    # First, collect features from standard ViT (no Hopfield)
    model_no_hopfield = deepcopy(model)
    for block in model_no_hopfield.backbone.blocks:
        if hasattr(block, 'hopfield_layer'):
            # Remove Hopfield influence
            block.hopfield_layer.memory_keys.data.zero_()
            block.hopfield_layer.memory_vals.data.zero_()
    
    # Extract features without Hopfield
    X_train_no_hop, y_train = extract_features(model_no_hopfield, train_loader)
    X_test_no_hop, y_test = extract_features(model_no_hopfield, test_loader)
    
    # Train baseline classifier
    clf_baseline = LogisticRegression(max_iter=1000)
    clf_baseline.fit(X_train_no_hop, y_train)
    baseline_acc = clf_baseline.score(X_test_no_hop, y_test)
    
    # Now test with increasing memory patterns
    results = []
    for n_patterns in range(10, max_patterns+1, 20):
        # Create a copy with limited memory
        model_test = deepcopy(model)
        
        # Limit memory size
        for block in model_test.backbone.blocks:
            if hasattr(block, 'hopfield_layer'):
                block.hopfield_layer.mem_size = n_patterns
                # Truncate memory
                block.hopfield_layer.memory_keys = nn.Parameter(
                    block.hopfield_layer.memory_keys[:, :n_patterns].clone()
                )
                block.hopfield_layer.memory_vals = nn.Parameter(
                    block.hopfield_layer.memory_vals[:, :n_patterns].clone()
                )
        
        # Extract features with limited memory
        X_train_hop, _ = extract_features(model_test, train_loader)
        X_test_hop, _ = extract_features(model_test, test_loader)
        
        # Train classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_hop, y_train)
        acc = clf.score(X_test_hop, y_test)
        
        results.append((n_patterns, acc))
        print(f"Memory size {n_patterns}: accuracy={acc:.4f} (baseline={baseline_acc:.4f})")
    
    # Plot results
    patterns, accs = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(patterns, accs, 'o-')
    plt.axhline(y=baseline_acc, color='r', linestyle='--', label='No Hopfield')
    plt.xlabel('Memory Size')
    plt.ylabel('Classification Accuracy')
    plt.title('Memory Capacity Test')
    plt.legend()
    plt.savefig('memory_capacity.png')
    plt.show()
    
    # Find optimal memory size
    optimal_idx = np.argmax(accs)
    print(f"Optimal memory size: {patterns[optimal_idx]} "
          f"(accuracy={accs[optimal_idx]:.4f})")
```

---

## 📊 Interpretation Guide: What to Look For

### Healthy Hopfield-Augmented ViT Characteristics:

| Metric | Healthy Range | Problem Indicators |
|--------|---------------|---------------------|
| **Memory Pattern Diversity** | Entropy > 0.7 | Low entropy (<0.3) = redundant patterns |
| **Memory Utilization** | 60-90% of slots used | <30% = underutilized; >95% = overcrowded |
| **Memory Specialization** | Some slots specialized (>0.6 score) | No specialization = memory not learning |
| **Temperature-Entropy Correlation** | Positive (>0.5) | Negative = unstable retrieval |
| **Query Subspace Dimensionality** | ~50-70% of full dim | Too low = loss of information |
| **Attention Pattern Similarity** | Moderate (0.4-0.6) | Too high = no added value; Too low = disruptive |

---

## 🚀 Pro Tips for Effective Analysis

1. **Start with the first Hopfield layer** - it has the most interpretable patterns
2. **Compare against a baseline** - always analyze a standard ViT alongside your model
3. **Track metrics during training** - don't just analyze the final model
4. **Focus on class tokens** - they contain the most semantic information
5. **Use domain-specific datasets** - medical images will show different patterns than natural images
6. **Combine quantitative and qualitative analysis** - numbers tell part of the story

---

## 💡 Key Insight for Vision Tasks

In vision applications, **memory patterns often specialize for visual concepts** rather than exact image copies. You'll typically see:

- Early layers: Low-level pattern detectors (edges, textures)
- Middle layers: Object parts and components
- Late layers: High-level semantic concepts

This hierarchical organization is a sign your Hopfield augmentation is working correctly. If all memory patterns look similar across layers, your integration likely has issues.

By implementing these analyses, you'll gain unprecedented insight into how your Hopfield-augmented ViT is actually working - knowledge that's essential for debugging, optimizing, and improving your model's performance on vision tasks.
