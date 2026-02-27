# Modern Hopfield Layer

## Overview

Modern Hopfield Networks (MHN) are continuous-state associative memory models that generalize classical Hopfield networks. They were introduced to address the severe storage capacity limitations of classical binary Hopfield networks, which could only store approximately 0.14N patterns for an N-node network.

The key breakthrough came from work by Krotov and Hopfield (2016) and Demircigil et al. (2017), who introduced new energy functions with exponential interactions that achieve **exponential storage capacity** — storing approximately exp(O(N)) patterns rather than just O(N).

Most remarkably, Ramsauer et al. (2020) demonstrated that the **update rule of modern Hopfield networks with continuous states is mathematically equivalent to the self-attention mechanism** in Transformers. This discovery established a deep theoretical connection between associative memory models and modern deep learning architectures.

## Energy Function

The energy function for continuous modern Hopfield networks is:

$$E(\xi) = -\text{lse}(\beta, \mathbf{X}^T \xi) + \frac{1}{2} \|\xi\|^2 + \beta^{-1}\log N + \frac{1}{2}M^2$$

where:
- $\mathbf{X} \in \mathbb{R}^{d \times M}$ is the matrix of M stored patterns
- $\xi \in \mathbb{R}^d$ is the query/state vector
- $\beta$ is the inverse temperature parameter (analogous to $1/\sqrt{d_k}$ in attention)
- $\text{lse}(\beta, z) = \beta^{-1}\log\sum_{i=1}^M e^{\beta z_i}$ is the log-sum-exp function

## Update Rule

The retrieval update is obtained by gradient descent on the energy function:

$$\xi^{\text{new}} = \mathbf{X} \cdot \text{softmax}(\beta \mathbf{X}^T \xi)$$

This is exactly the **scaled dot-product attention** mechanism:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where:
- $Q = \xi^T$ (query)
- $K = \mathbf{X}$ (keys = stored patterns)
- $V = \mathbf{X}$ (values = stored patterns)
- $\beta = 1/\sqrt{d_k}$ (scaling factor)

## Key Properties

1. **Exponential Storage Capacity**: Can store approximately exp(c·d) patterns in d-dimensional space

2. **One-Step Convergence**: Retrieves patterns in a single update step with exponentially small error

3. **Continuous States**: Generalizes from binary to continuous-valued patterns and states

4. **Differentiable**: Can be integrated into deep learning architectures

5. **Three Types of Energy Minima**:
   - Global fixed point: averaging over all patterns
   - Metastable states: averaging over subsets of similar patterns
   - Fixed points: retrieval of individual patterns

## Connection to Transformers

The correspondence between MHN and Transformer attention:

| Modern Hopfield Network | Transformer Attention |
|------------------------|----------------------|
| Query vector $\xi$ | Query $Q$ |
| Stored patterns $\mathbf{X}$ | Keys $K$ and Values $V$ |
| Inverse temperature $\beta$ | Scaling $1/\sqrt{d_k}$ |
| Energy minimization | Attention computation |
| Pattern retrieval | Context vector computation |

The **adiabatic approximation** (where hidden states evolve slowly) makes this correspondence exact. Recent work by Masumura and Taki (2025) extends this relationship beyond the adiabatic approximation by adding hidden states to self-attention, showing improvements for rank collapse and token uniformity problems.

## Hopfield Layers

Practical implementations provide several layer types:

- **HopfieldLayer**: General associative memory lookup
- **HopfieldPooling**: Pooling operation using Hopfield attention
- **HopfieldNetwork**: Full recurrent Hopfield network

These can be integrated into deep learning architectures for:
- Memory-augmented models
- Set pooling operations
- Associative retrieval tasks
- Immune repertoire classification

## References

1. Ramsauer, H., et al. (2020). "Hopfield Networks is All You Need." arXiv:2008.02217
2. Krotov, D., & Hopfield, J. (2016). "Dense Associative Memory for Pattern Recognition." NeurIPS
3. Demircigil, M., et al. (2017). "On A Model of Associative Memory With Huge Storage Capacity." arXiv:1702.01929
4. Masumura, T., & Taki, M. (2025). "On the Role of Hidden States of Modern Hopfield Network in Transformer." arXiv:2511.20698
5. Hu, J. Y.-C., et al. (2024). "Outlier-Efficient Hopfield Layers for Large Transformer-Based Models." ICML
