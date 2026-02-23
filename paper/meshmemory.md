## Abstract

Abstract Recursive transformers reuse parameters and iterate over hidden states multiple times, decoupling compute depth from parameter depth.
However, under matched compute, recursive models with fewer parameters often lag behind non-recursive counterparts.
By probing hidden states, we trace this performance gap to two primary bottlenecks: undifferentiated computation , where the core is forced to adopt a similar computational pattern at every iteration, and information overload , where long-lived and transient information must coexist in a single hidden state.
To address the issues, we introduce a Me mory-as- S tate- H ighways (MeSH) scheme, which externalizes state management into an explicit memory buffer and employs lightweight routers to dynamically diversify computation across iterations.
Probing visualizations confirm that MeSH successfully resolves the pathologies by inducing functional specialization across iterations. On the Pythia suite (160M–1.4B), MeSH-enhanced recursive transformers consistently improve over recursive baselines and outperforms its larger non-recursive counterpart at the 1.4B scale, improving average downstream accuracy by +1.06% with 33% fewer non-embedding parameters. Our analysis establishes MeSH as a scalable and principled architecture for building stronger recursive models.

## 1 Introduction

Scaling up model parameters and data has been a primary driver of improvements in the general capabilities of large language models (LLMs)
(Kaplan et al., 2020; Hoffmann et al., 2022; Brown et al., 2020; Wei et al., 2022; Chowdhery et al., 2023; Grattafiori et al., 2024; OpenAI, 2023; Snell et al., 2024; Liu et al., 2024; Comanici et al., 2025)
. However, further gains along this axis face headwinds: the supply of high-quality text is nearing exhaustion
(Villalobos et al., 2022; Muennighoff et al., 2023)
, empirical scaling curves show signs of saturation
(Hackenburg et al., 2025; Hoffmann et al., 2022), and distributed pre-training incurs substantial, often super-linear, communication overheads as models grow
(Narayanan et al., 2021; Pati et al., 2023; Li et al., 2024; Patterson et al., 2021; Momeni et al., 2025).

As a parameter‑efficient architectural response to the scaling bottlenecks of large models, recursive transformers have recently attracted growing interest (Geiping et al., 2025; Bae et al., 2024; 2025; Zeng et al., 2025; Saunshi et al., 2025). The core idea behind is to decouple computational depth from parameter depth by repeatedly applying a compact, weight‑shared core block in a loop. By breaking the tight coupling between these two depths, recursive transformers natively enable dynamic computation: they can, in principle, allocate computational budgets adaptively based on task difficulty to reduce inductive bias (Bae et al., 2025), and open up a new scaling axis of computational depth, complementing model size and data volume (Zhu et al., 2025b; Geiping et al., 2025; Saunshi et al., 2025).

However, a critical challenge remains: under matched compute, recursive models with fewer parameters often lag behind their non-recursive counterparts. In this work, we provide measurable evidence that the performance gap stems from fundamental bottlenecks:
undifferentiated computation and information overload, quantified by three observables as skewed computation, representational stagnation, and dimensional collapse. To address the pathologies, we propose the Memory-as-State-Highways (MeSH) scheme, a principled architectural modification that replaces the overloaded hidden state with an explicit memory buffer governed by lightweight, step-wise routers.
The proposed design separates persistent memory from transient computation, effectively converting the implicit challenge of state management into a clear, learnable routing problem for recursive transformers.

## 2 Why Naive Recursion Fails: A Diagnostic Analysis

Figure: Figure 1: Diagnostic visualizations of the recursive transformer. Analyses are performed on a Pythia-410M-based model with the Prelude-Reccurent-Coda architecture (3 core loops). Hidden state matrices ($\mathbf{h}\in\mathbb{R}^{\text{seq}\times\text{dim}}$) are extracted from 500 samples from the Pile dataset. The states $\mathbf{h}_{\text{emb}},\mathbf{h}^{(0)}\dots\mathbf{h}_{\text{out}}$ refer to the initial token embeddings, the states to each block, and the final output state. We leave further experimental details and analysis to Section [4.1](https://arxiv.org/html/2510.07739v1#S4.SS1). (a) Skewed computational pattern. Plots the relative magnitude of the state update, calculated for each computational block ($f$) as $2||f(\mathbf{h})-\mathbf{h}||_{F}/(||f(\mathbf{h})||_{F}+||\mathbf{h}||_{F})$, where $||\cdot||_{F}$ is the Frobenius norm, which serves as a proxy for the computational effort of each block. Bars show the mean and standard deviation across 500 samples. (b) Representational stagnation. Displays the pairwise Centered Kernel Alignment (CKA) (Kornblith et al., 2019) similarity with an RBF kernel between the hidden state matrices. (c) Loop representational collapse. Shows the top 50 normalized singular values ($\sigma_{i}/\sigma_{0}$) for each hidden state matrix on a logarithmic Y-axis. The decay rate of the spectrum indicates the effective rank or intrinsic dimensionality of each state matrix.
Refer to caption: x1.png

The core premise of a recursive transformer is to reuse a weight-shared computational block, yet the design introduces a fundamental limitation: the block lacks any explicit information about its progress within the iterative sequence, which prevents effective functional specialization and leads to inefficient computation.
This also forces a single hidden state to handle multiple conflicting information. We define these two primary bottlenecks as undifferentiated computation and information overload.

#### Undifferentiated computation.

The inability to differentiate between loop steps prevents the model from assigning specialized roles to each iteration. This leads to two distinct failure modes. First, the model exhibits a skewed computational pattern, as shown in Figure [1](https://arxiv.org/html/2510.07739v1#S2.F1)a. The first core loop performs the vast majority of the computational work, while subsequent iterations contribute negligibly. This indicates the model fails to distribute its process over multiple steps. Second, this lack of directed computation results in representational stagnation (Figure [1](https://arxiv.org/html/2510.07739v1#S2.F1)b). High CKA similarity between consecutive loop states reveals that the model becomes trapped in a fixed-point attractor, repeatedly applying a near-identical transformation instead of progressively refining its representation.

#### Information overload.

Concurrently, the single hidden state vector is forced to be the sole carrier for all information, creating a severe bottleneck, where the single state could be forced to manage multiple, often conflicting, roles simultaneously:

- •
Long-term Memory: Preserving key information from the initial input to prevent catastrophic forgetting.
- •
Working Memory: Preparing intermediate features for the subsequent iteration.
- •
Final Output: Exposing features that are immediately useful for the final prediction layer.

The information overload on the hidden state forces the model to find a low-dimensional “common ground” representation that can safely survive multiple transformations, which directly causes loop representational collapse. We quantify this by analyzing the normalized singular value spectrum of the hidden state matrices, a proxy for their effective rank (see Figure [1](https://arxiv.org/html/2510.07739v1#S2.F1)c). The singular value spectrum of the loop states decays much more rapidly than that of the initial state, indicating a collapse into a lower-dimensional subspace and a significant loss of expressive capacity.

The diagnosis of undifferentiated computation and information overload directly motivates our solution, MeSH, which is specifically designed to address these identified problems.

## 3 Methodology: Alleviating Information Overload and Enabling Functional Specialization

The pathologies diagnosed in Section [2](https://arxiv.org/html/2510.07739v1#S2)—undifferentiated computation and information overload—stem from the architectural limitations of naive recursion. In this section, we develop a methodology aimed at alleviating these core issues. We first review common heuristic-based recurrence schemes, which use fixed, additive connections to supplement the context at each step, that can be seen as attempts to mitigate information overload but do little to address the problem of undifferentiated computation. We then introduce our proposed solution, MeSH, a general framework designed to systemically alleviate both information overload and the lack of functional specialization.

### 3.1 Preliminaries: Architecture of Recursive Transformers

#### Overall Architectural Structure.

Recursive transformers achieve computational depth by repeatedly applying a shared, weight-tied core block, $f_{\text{core}}(\cdot)$. The central idea is to refine a hidden state $\mathbf{h}^{(t)}\in\mathbb{R}^{L\times D}$ over a sequence of $K$ iterations. Starting from an initial state $\mathbf{h}^{(0)}$, the simplest form of recurrence updates the state as $\mathbf{h}^{(t+1)}=f_{\text{core}}(\mathbf{h}^{(t)})$.
The core recurrence loop could be embedded within a broader network topology that defines how the initial state $\mathbf{h}^{(0)}$ is produced and how the final state $\mathbf{h}^{(K)}$ is consumed. We adopt the Prelude-Recurrent-Coda structure (Geiping et al., 2025) (also called Middle-Cycle (Bae et al., 2025)), which augments the core recursive block with specialized, non-tied prelude and coda networks. The framework first uses a prelude block, $f_{\text{pre}}$, to process the initial token embeddings ($\mathbf{h}_{\text{emb}}$) and prepare the first state for the loop: $\mathbf{h}^{(0)}=f_{\text{pre}}(\mathbf{h}_{\text{emb}})$. The recursive loop then runs for $K$ steps, after which its final output state, $\mathbf{h}^{(K)}$, is passed to a coda block, $f_{\text{coda}}$, to produce the model’s final representation: $\mathbf{h}_{\text{final}}=f_{\text{coda}}(\mathbf{h}^{(K)})$.

#### Core Recurrence Variants.

While the base recurrence, $\mathbf{h}^{(t+1)}=f_{\text{core}}(\mathbf{h}^{(t)})$, represents a straightforward cascade of computations, it may struggle with information retention, as each iteration can overwrite or forget crucial aspects of its input. To alleviate this representational burden on the core, the state update can be augmented with historical information. We summarize several common variants, which represent different strategies for information propagation (illustrated in Figure  [2](https://arxiv.org/html/2510.07739v1#S3.F2)). The general update rule with such a context supplement is:

$$ $\mathbf{h}^{(t+1)}=f_{\text{core}}(\mathbf{h}^{(t)})+\mathbf{h}_{\text{sup}}^{(t)}$ (1) $$

where $\mathbf{h}_{\text{sup}}^{(t)}$ is a supplementary context. Common choices for this context include:

- •
Residual: Setting $\mathbf{h}_{\text{sup}}^{(t)}=\mathbf{h}^{(t)}$ introduces a standard skip connection between adjacent iterations, which allows the core to learn a residual update, enabling the model to incrementally refine the representation and aggregate information from all preceding steps (Yu et al., 2025; Zeng et al., 2025; Bae et al., 2025).
- •
Anchor: Setting $\mathbf{h}_{\text{sup}}^{(t)}=\mathbf{h}^{(0)}$ explicitly tethers each iteration to the initial state that entered the loop. The intuition is to prevent the iterative process from drifting too far from the initial semantics by continuously reinforcing the starting context (Yang et al., 2023; Mohtashami et al., 2023; Geiping et al., 2025).
- •
Anchor*: An alternative, $\mathbf{h}_{\text{sup}}^{(t)}=\mathbf{h}_{\text{emb}}$, anchors each iteration to the raw token embeddings.

The heuristic connectivity schemes can be seen as attempts to mitigate information overload. By providing a direct, additive path for historical information (like the initial state $\mathbf{h}^{(0)}$ or the previous state $\mathbf{h}^{(t)}$), they partially offload the burden of memory from the main hidden state pathway. This allows the core to focus more on transformation rather than just preservation. However, it is important to note that these are rigid, non-adaptive solutions. The choice among them is often a heuristic design decision that requires careful empirical validation. Crucially, they do little to address the problem of undifferentiated computation, as the core block remains blind to its position in the loop.

Figure: Figure 2: Comparison of recurrence schemes. (a) The general architecture of a recursive transformer involves the general dataflow passing a state $\mathbf{h}^{(t)}$ through a core computational block $f_{\text{core}}$ to produce the next state $\mathbf{h}^{(t+1)}$. (b) Common heuristic variants employ a fixed, additive state update to optimize the information flow, where the core output is supplemented by historical states $\mathbf{h}_{\text{sup}}$ (e.g., initial state $\mathbf{h}^{(0)}$ for anchor or previous state $\mathbf{h}^{(t)}$ for residual). (c) Our proposed MeSH replaces this rigid addition with a dynamic memory mechanism, which explicitly manages historical states via learnable write and read operations, allowing the model to flexibly retrieve and combine information to form the next state $\mathbf{h}^{(t+1)}$.
Refer to caption: x2.png

### 3.2 MeSH: Memory–as-State-Highways for Recursive Transformers

We move beyond fixed recurrence rules by introducing the Memory–as-State-Highways (MeSH), a mechanism that replaces the simple state-passing scheme. As shown in Figure [2](https://arxiv.org/html/2510.07739v1#S3.F2), MeSH externalizes state management into an explicit state buffer controlled by learnable, step-wise read-write routers. This design decouples transient computation within the recursive core from persistent memory, allowing the model to dynamically manage iteration-specific information flow. The MeSH-augmented loop consists of several lightweight components:

#### 1. State Buffer and Initialization.

MeSH maintains a state buffer $\mathbf{M}$ with $B$ memory slots, $\mathbf{M}=\{\mathbf{m}_{0},\mathbf{m}_{1},\ldots,\mathbf{m}_{B-1}\}$, where each slot $\mathbf{m}_{b}\in\mathbb{R}^{L\times D}$ shares the same dimensions as the hidden states. Before the loop begins, the buffer is initialized by placing the raw token embeddings, $\mathbf{h}_{\text{emb}}$, into the first slot. This designated slot, $\mathbf{m}_{0}$, serves as a initial anchor to the original input. All other slots are initialized to zero:

$$ $\mathbf{m}_{0}^{(0)}=\mathbf{h}_{\text{emb}},\quad\text{and}\quad\mathbf{m}_{b>0}^{(0)}=0$ (2) $$

#### 2. Core Computation and Dynamic Routers.

The core block $f_{\text{core}}$ remains the central computational unit. The interface to the buffer is managed by step-wise Write and Read Routers ($R_{\text{write}}^{(t)}$ and $R_{\text{read}}^{(t)}$), which have unique parameters for each iteration $t=0,\dots,K-1$. At each step, they compute routing weights based on the current hidden state $\mathbf{h}^{(t)}\in\mathbb{R}^{L\times D}$:

$$ $\mathbf{w}_{\text{write}}^{(t)}=\text{Softmax}(\text{Linear}_{\text{write}}^{(t)}(\mathbf{h}^{(t)})),\quad\mathbf{w}_{\text{read}}^{(t)}=\text{Softmax}(\text{Linear}_{\text{read}}^{(t)}(\mathbf{h}^{(t)}))$ (3) $$

Each $\text{Linear}^{(t)}$ function is a one-layer projection that maps the $D$-dimensional hidden state of each token to a vector of $B$ logits, corresponding to the number of buffer slots. A softmax function is then applied along the slot dimension for each token to normalize these logits, producing the final weight matrices $\mathbf{w}_{\text{write}}^{(t)}$ and $\mathbf{w}_{\text{read}}^{(t)}$, both of shape $\mathbb{R}^{L\times B}$.

#### 3. MeSH-Augmented Recurrence and Integration.

The fixed context supplementation is replaced by a memory update logic, as illustrated in Figure [2](https://arxiv.org/html/2510.07739v1#S3.F2)b. At each step $t$, the core first computes its output $\mathbf{h}_{\text{m}}^{(t)}$ from the current state $\mathbf{h}^{(t)}$:

$$ $\mathbf{h}_{\text{m}}^{(t)}=f_{\text{core}}(\mathbf{h}^{(t)})$ (4) $$

The buffer is then updated via a distributed write operation for a soft insertion of the state, where the output $\mathbf{h}_{\text{m}}^{(t)}$ is scaled by the computed write weights before being added to the memory slot:

$$ $\mathbf{m}_{b}^{(t+1)}=\mathbf{m}_{b}^{(t)}+\mathbf{h}_{\text{m}}^{(t)}\odot\mathbf{w}_{\text{write},b}^{(t)},\quad\text{for }b=0,\dots,B-1$ (5) $$

where $\odot$ denotes element-wise multiplication with broadcasting. Subsequently, the state for the next iteration, $\mathbf{h}^{(t+1)}$, is synthesized via a read operation from the updated buffer:

$$ $\mathbf{h}^{(t+1)}=\sum_{b=0}^{B-1}\mathbf{m}_{b}^{(t+1)}\odot\mathbf{w}_{\text{read},b}^{(t)}$ $$

In the prelude-recurrent-coda setting, a dedicated transitional write-read cycle first processes the prelude’s output $f_{\text{pre}}(\mathbf{h}_{\text{emb}})$ to synthesize the initial state $\mathbf{h}^{(0)}$. After the main loop, a final read operation computes the output $\mathbf{h}^{(K)}$ from the memory buffer before passed to the coda. The full computational process is detailed in the pseudocode in Appendix [C](https://arxiv.org/html/2510.07739v1#A3).

### 3.3 How MeSH Addresses the Diagnosed Pathologies

The architectural design of MeSH, centered on state externalization and dynamic routing, directly counteracts the core pathologies diagnosed in Section [2](https://arxiv.org/html/2510.07739v1#S2).

#### Enabling Functional Specialization via Dynamic State Composition.

MeSH explicitly breaks the cycle of undifferentiated computation by replacing the rigid, additive update rule of heuristic methods with a dynamic read-write cycle controlled by step-wise routers. Since each router ($R_{\text{write}}^{(t)},R_{\text{read}}^{(t)}$) has its own unique set of learnable parameters for each iteration $t$, the model is no longer forced to apply a single, universal transformation. Instead, at each step, it learns to dynamically synthesize the next state by retrieving a context-specific mixture of information from the memory buffer, which contains all relevant historical states. This flexibility allows MeSH to learn and dynamically switch between complex recurrence behaviors. The ability to adapt the recurrence rule on the fly is the implicit mechanism to enables functional specialization.

#### Alleviating Information Overload via State Externalization.

MeSH directly alleviates information overload by decoupling persistent memory from transient computation. The external state buffer $\mathbf{M}$ serves as a dedicated, multi-slot highway for long-lived information. This design relieves the primary hidden state $\mathbf{h}^{(t)}$ from the burden of simultaneously storing historical context and serving as the workspace for the core block. The hidden state can now utilize its full dimensionality for complex, transient computations, knowing that essential long-term information is safely preserved in the buffer and can be retrieved on demand by the read router. This allows the model to maintain high-dimensional, expressive representations throughout the entire iterative process.

In essence, MeSH replaces the single, overloaded information channel of standard recurrence with a multi-slot memory buffer and dynamic, state-aware routers. The principled design provides a systemic and highly expressive solution to the core problems in recursive transformers, subsuming prior heuristic approaches into a more general framework (see Appendix [D](https://arxiv.org/html/2510.07739v1#A4) for more discussion).

## 4 Experiments

We pretrain our models from scratch, closely following the methodology of the Pythia suite (Biderman et al., 2023). We employ the same GPT-NeoX-based architecture and train on a deduplicated subset of the Pile dataset (Gao et al., 2020), curated by EleutherAI. For evaluation, we assess two primary aspects of model performance. We report perplexity scores on the validation sets of the Pile (Gao et al., 2020), Wikitext, and the Lambada (both OpenAI and Standard versions) datasets (Paperno et al., 2016) to measure language modeling capabilities. We also evaluate downstream performance on a suite of 9 few-shot benchmarks using the LM Evaluation Harness framework (Gao et al., 2024). Detailed training configurations and evaluation procedures are described in Appendix [B](https://arxiv.org/html/2510.07739v1#A2).

### 4.1 A Comparative Diagnostic Analysis of Recurrence Schemes

In Section [2](https://arxiv.org/html/2510.07739v1#S2), we identified three critical symptoms arising from naive recursive transformers: a skewed computational pattern, representational stagnation, and loop representational collapse. We conduct a detailed analysis of the internal dynamics of four model variants: a base recursive model, two common heuristic variants (+residual and +anchor), and our proposed +mesh architecture. The analysis is performed on a Pythia-410M model with configuration of 3+6R3+3, averaging results over 500 samples from the Pile dataset.

Figure: Figure 3: Skewed Computational Pattern. Plots the relative magnitude of the state update, calculated for each computational block ($f$) as $2||f(\mathbf{h})-\mathbf{h}||_{F}/(||f(\mathbf{h})||_{F}+||\mathbf{h}||_{F})$, where $||\cdot||_{F}$ is the Frobenius norm, which serves as a proxy for the computational effort of each block. Bars show the mean and standard deviation across 500 samples.
Refer to caption: x3.png

#### MeSH mitigates the skewed computational pattern.

Figure [3](https://arxiv.org/html/2510.07739v1#S4.F3) visualizes the computational effort of each block, confirming the pathology of naive recursion. The base model exhibits an extreme computational imbalance: the first core loop ($1^{\text{st}}f_{\text{core}}$) accounts for the vast majority of the work, while subsequent loops contribute negligibly, demonstrating a classic case of diminishing returns. While the +residual and +anchor heuristics offer partial relief, the computational effort still decays sharply. In stark contrast, the +mesh model achieves a remarkably balanced computational distribution, with all three core loops contributing significantly. This demonstrates that MeSH’s dynamic read-write mechanism endows the model with a sense of iterative progress, allowing it to assign distinct and meaningful computational roles to each step.

Figure: Figure 4: Representational Stagnation. Displays the pairwise Centered Kernel Alignment (CKA) similarity with an RBF kernel between hidden state matrices ($\mathbf{h}\in\mathbb{R}^{\text{seq}\times\text{dim}}$) at different stages of the model. The matrix shows the mean similarity across 500 samples. High similarity (values near 1.0) between consecutive loop states indicates that representations have stopped evolving.
Refer to caption: x4.png

#### MeSH breaks representational stagnation.

Figure [4](https://arxiv.org/html/2510.07739v1#S4.F4) displays the CKA similarity (Kornblith et al., 2019) between hidden states. High similarity between consecutive loop states ($h^{(1)}$, $h^{(2)}$, $h^{(3)}$) signals that the model is trapped in a fixed-point attractor. The base model’s loop states exhibit very high CKA similarity, confirming severe representational stagnation.
The +mesh model reduces the similarity between consecutive loop states, proving it has broken free from stagnation while its memory buffer allows it to maintain a strong connection to the initial context.

Figure: Figure 5: Loop Representational Collapse. Shows the top 50 normalized singular values ($\sigma_{i}/\sigma_{0}$) for key hidden state matrices on a logarithmic Y-axis. The decay rate of the spectrum indicates the effective rank or intrinsic dimensionality of each state matrix. A faster decay signifies a collapse into a lower-dimensional representation. Lines and shaded areas represent the mean and standard deviation across 500 samples.
Refer to caption: x5.png

#### MeSH prevents loop representational collapse.

Figure [5](https://arxiv.org/html/2510.07739v1#S4.F5) plots the singular value spectrum for hidden states. In the base model, the loop states ($h^{(1)}$, $h^{(2)}$, $h^{(3)}$) show a much faster spectral decay than the input state ($h^{(0)}$), confirming loop representational collapse into a low-dimensional subspace as a result of a forced “representational compromise”. The heuristic fixes offer only marginal gains. The +mesh model, however, demonstrates the ability to preserve representational richness, allowing the hidden state to maintain a high-dimensional, expressive structure throughout the iterative process.

### 4.2 Main Results

**Table 1: Comparison of MeSH, Recursive and Vanilla Transformers. Performance is measured by perplexity (PPL$\downarrow$) on four datasets and average accuracy (Avg. acc$\uparrow$) on a suite of 10 downstream tasks. The percentage reduction in non-embedding parameters for recursive models is shown in parentheses. The Layers for recursive models follow the format ‘$\{L_{\text{prelude}}\}$+$\{L_{\text{core}}\}\text{R}\{N_{\text{loop}}\}$+$\{L_{\text{coda}}\}$’, indicating the number of layers in the prelude, core, coda. $\Delta\text{acc}$ shows the absolute accuracy change relative to the Vanilla baseline. LD-O and LD-S refer to Lambada OpenAI and Standard. The best results within each recursive block are bolded, and second best results are underlined.**
|  | Structure | Perplexity↓ | Task Avg. acc↑ / $\Delta$acc (%) |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Scheme | Layers | Variant | Pile | Wiki | LD-O | LD-S | 0-shot | 5-shot |
| 160M | Vanilla | 12 | — | \cellcolorVanillaC11.31 | \cellcolorVanillaC30.32 | \cellcolorVanillaC42.86 | \cellcolorVanillaC129.89 | \cellcolorVanillaC39.88 | \cellcolorVanillaC40.54 |
| Recursive<br>(-33.3%) | 2+4R2+2 | \cellcolorBaseCbase | \cellcolorBaseC11.79 | \cellcolorBaseC32.32 | \cellcolorBaseC53.06 | \cellcolorBaseC217.87 | \cellcolorBaseC38.90 / -0.98 | \cellcolorBaseC39.29 / -1.25 |  |
| +anchor | 11.63 | 31.69 | 50.38 | 195.11 | 38.81 / -1.07 | 40.15 / -0.39 |  |  |  |
| \cellcolormeshgreen+mesh | \cellcolormeshgreen11.37 | \cellcolormeshgreen30.43 | \cellcolormeshgreen46.60 | \cellcolormeshgreen178.77 | \cellcolormeshgreen39.41 / -0.47 | \cellcolormeshgreen40.60 / +0.06 |  |  |  |
| Pythia-410M | Vanilla | 24 | — | \cellcolorVanillaC9.07 | \cellcolorVanillaC21.79 | \cellcolorVanillaC19.48 | \cellcolorVanillaC65.86 | \cellcolorVanillaC43.87 | \cellcolorVanillaC45.31 |
| Recursive<br>(-33.3%) | 4+8R2+4 | \cellcolorBaseCbase | \cellcolorBaseC9.31 | \cellcolorBaseC22.74 | \cellcolorBaseC22.57 | \cellcolorBaseC53.76 | \cellcolorBaseC43.26 / -0.61 | \cellcolorBaseC45.03 / -0.28 |  |
| +anchor | 9.19 | 22.12 | 20.37 | 52.55 | 43.70 / -0.17 | 45.68 / +0.37 |  |  |  |
| \cellcolormeshgreen+mesh | \cellcolormeshgreen9.09 | \cellcolormeshgreen21.84 | \cellcolormeshgreen19.63 | \cellcolormeshgreen42.51 | \cellcolormeshgreen44.12 / +0.25 | \cellcolormeshgreen45.60 / +0.29 |  |  |  |
| Recursive<br>(-50.0%) | 3+6R3+3 | \cellcolorBaseCbase | \cellcolorBaseC9.65 | \cellcolorBaseC23.88 | \cellcolorBaseC26.76 | \cellcolorBaseC81.75 | \cellcolorBaseC41.94 / -1.93 | \cellcolorBaseC44.01 / -1.30 |  |
| +residual | 9.69 | 24.05 | 26.31 | 76.76 | 42.16 / -1.71 | 44.24 / -1.07 |  |  |  |
| +anchor | 9.49 | 23.31 | 24.49 | 72.30 | 42.85 / -1.02 | 44.90 / -0.41 |  |  |  |
| \cellcolormeshgreen+mesh | \cellcolormeshgreen9.35 | \cellcolormeshgreen22.80 | \cellcolormeshgreen20.72 | \cellcolormeshgreen52.07 | \cellcolormeshgreen43.53 / -0.34 | \cellcolormeshgreen46.04 / +0.73 |  |  |  |
| Pythia-1B | Vanilla | 16 | — | \cellcolorVanillaC7.96 | \cellcolorVanillaC17.66 | \cellcolorVanillaC13.53 | \cellcolorVanillaC33.65 | \cellcolorVanillaC46.95 | \cellcolorVanillaC49.07 |
| Recursive<br>(-31.3%) | 3+5R2+3 | \cellcolorBaseCbase | \cellcolorBaseC8.20 | \cellcolorBaseC18.64 | \cellcolorBaseC14.44 | \cellcolorBaseC36.39 | \cellcolorBaseC45.72 / -1.23 | \cellcolorBaseC47.75 / -1.32 |  |
| +residual | 8.19 | 18.46 | 14.18 | 35.54 | 46.19 / -0.76 | 47.85 / -1.22 |  |  |  |
| +anchor* | 8.07 | 18.06 | 12.90 | 30.56 | 46.85 / -0.10 | 49.18 / +0.11 |  |  |  |
| +anchor | 8.10 | 18.15 | 13.32 | 32.34 | 46.73 / -0.22 | 48.83 / -0.24 |  |  |  |
| \cellcolormeshgreen+mesh | \cellcolormeshgreen7.90 | \cellcolormeshgreen17.54 | \cellcolormeshgreen12.19 | \cellcolormeshgreen26.71 | \cellcolormeshgreen47.53 / +0.58 | \cellcolormeshgreen49.51 / +0.44 |  |  |  |
| Pythia-1.4B | Vanilla | 24 | — | \cellcolorVanillaC7.44 | \cellcolorVanillaC15.97 | \cellcolorVanillaC10.51 | \cellcolorVanillaC22.81 | \cellcolorVanillaC49.50 | \cellcolorVanillaC51.93 |
| Recursive<br>(-33.3%) | 4+8R2+4 | \cellcolorBaseCbase | \cellcolorBaseC7.63 | \cellcolorBaseC16.64 | \cellcolorBaseC11.38 | \cellcolorBaseC23.69 | \cellcolorBaseC48.89 / -0.61 | \cellcolorBaseC50.99 / -0.94 |  |
| +residual | 7.58 | 16.44 | 10.91 | 20.44 | 49.50 / +0.00 | 51.18 / -0.75 |  |  |  |
| +anchor* | 7.51 | 16.27 | 10.81 | 19.14 | 49.29 / -0.21 | 51.83 / -0.10 |  |  |  |
| +anchor | 7.51 | 16.25 | 10.71 | 19.37 | 49.39 / -0.11 | 51.27 / -0.66 |  |  |  |
| \cellcolormeshgreen+mesh | \cellcolormeshgreen7.39 | \cellcolormeshgreen15.84 | \cellcolormeshgreen9.72 | \cellcolormeshgreen19.39 | \cellcolormeshgreen50.56 / +1.06 | \cellcolormeshgreen52.79 / +0.86 |  |  |  |

We conducted experiments on Pythia models ranging from 160M to 1.4B parameters, creating recursive variants with approximately 33% fewer non-embedding parameters to compare against standard Vanilla baselines and simpler recursive schemes (Table [1](https://arxiv.org/html/2510.07739v1#S4.T1)). While naive recursion (base) degrades performance and fixed schemes like +anchor offer only partial recovery, our MeSH-enhanced models (+mesh) consistently outperform all other variants. The +mesh models can even surpass their larger, more parameter-heavy Vanilla counterparts. For instance, the Pythia-1.4B +mesh model, despite its smaller footprint, improves 0-shot and 5-shot average accuracy by +1.06% and +0.86% respectively over the Vanilla version, while also achieving state-of-the-art perplexity scores across all datasets. Furthermore, the performance advantage scales favorably with model size, confirming that MeSH’s dynamic state management is not only effective but also a highly efficient and scalable architectural principle.

### 4.3 Further Analysis and Ablation Studies

Figure: Figure 6: Training Dynamics of Recursive Variants. Comparison of training loss and downstream 0-shot accuracy for the 1.4B-Pythia-based recursive models. (Left) Training loss curve over 120k steps on a logarithmic x-axis. (Right) Downstream average 0-shot accuracy evaluated at checkpoints along a linear x-axis.
Refer to caption: x6.png

Analysis of Training Dynamics. To understand not just the final performance but also the learning process itself, we visualize the training dynamics of the 1.4B-parameter recursive variants in Figure [6](https://arxiv.org/html/2510.07739v1#S4.F6). By juxtaposing pre-training loss with downstream accuracy evaluated at various checkpoints, we can assess both the learning efficiency and the rate at which models acquire useful capabilities.
The training loss curves (Figure [6](https://arxiv.org/html/2510.07739v1#S4.F6), left panel) reveal that the +mesh model consistently achieves a lower loss throughout pre-training. This indicates superior learning efficiency, as MeSH is able to fit the training data more effectively at every stage compared to the base, +residual, and +anchor variants.
The training advantage translates directly into stronger downstream performance. The right panel of Figure [6](https://arxiv.org/html/2510.07739v1#S4.F6) shows that the +mesh model not only starts from a stronger initial checkpoint but also exhibits a steeper and more consistent improvement in 0-shot accuracy.
The superiority provides compelling evidence that MeSH’s architectural modifications fundamentally enhance the model’s ability to acquire and retain useful knowledge throughout the entire pre-training process, rather than being just a final-step improvement.

Figure: Figure 7: Scaling Analysis of MeSH vs. Baselines. Performance of Vanilla (non-recursive), naive Recursive, and +mesh models plotted against non-embedding parameter counts. (Left) Average downstream accuracy (0-shot and 5-shot). (Right) Evaluation loss.
Refer to caption: x7.png

Scaling Properties and Parameter Efficiency. We provide scaling results in Figure [7](https://arxiv.org/html/2510.07739v1#S4.F7), revealing the parameter efficiency of the MeSH architecture. While naive recursive models (blue lines) consistently underperform their standard vanilla counterparts (green lines) despite saving about 33% of parameters, our +mesh models (purple lines) not only decrease the performance degradation but could even outperform the Vanilla baselines at large scales. For example, our 805M-parameter +mesh model achieves 50.6% (0-shot) and 52.8% (5-shot) accuracy, surpassing the 1.2B-non-emb-parameter Vanilla model’s 49.5% (0-shot) and 51.9% (5-shot),
which translates to a 1.46x improvement in parameter efficiency, allowing a MeSH-enhanced model to achieve the same level of evaluation loss as a Vanilla model with almost a third fewer parameters.

Figure: Figure 8: PPL vs. Parameter Efficiency for Pythia-410M. The plot shows the Pile perplexity as a function of non-embedding parameters, shown as a percentage relative to the Vanilla baseline. Each point represents a different distribution of layers (prelude, core, coda). The total computational depth for all models is aligned with the 24-layer non-recursive Vanilla.
Refer to caption: x8.png

Impact of Layer Distribution and Parameter Scaling.
To further dissect the architectural benefits of MeSH, we conduct a control study on the distribution of layers within the prelude-loop-coda framework. Using the Pythia-410M architecture as a testbed, we train several recursive models with varying configurations while keeping the total compute equivalent to the 24-layer non-recursive Vanilla model. We plot the validation perplexity against the percentage of non-embedding parameters relative to the Vanilla baseline, with the results shown in Figure [8](https://arxiv.org/html/2510.07739v1#S4.F8).
the +mesh architecture (purple line) consistently achieves lower perplexity than the baseline recursive model (blue line) across all parameter allocations, demonstrating its robust performance advantage.
MeSH also shows remarkable parameter efficiency against the non-recursive baseline.
As a trend, the performance of +mesh (purple line) approaches that of the full 24-layer Vanilla model (green dashed line) while using approximately 30% fewer non-embedding parameters.
The study shows that MeSH is not just an additive improvement but a powerful architectural principle that enhances the parameter efficiency and scaling properties of recursive transformers.

## 5 Conclusion

In this work, we diagnose the underperformance of recursive transformers, tracing it, through the lens of
quantified observables, to the systemic pathologies of undifferentiated computation and information overload. We further propose MeSH as a principled architectural solution that externalizes state management into an explicit memory buffer controlled by dynamic, step-wise routers. Our experiments validate that MeSH successfully addresses the diagnosed pathologies while also delivering substantial performance gains on recursive backbones. We conclude that this work establishes explicit, routed state management as a scalable and effective principle for building stronger recursive models, offering a promising architectural path forward as the field seeks more sustainable scaling paradigms.

## References

- Aleksandrov et al. (2025)
Preslav Aleksandrov, Meghdad Kurmanji, Fernando Garcia Redondo, David O’Shea, William Shen, Alex Iacob, Lorenzo Sani, Xinchi Qiu, Nicola Cancedda, and Nicholas D Lane.
Abbie: Autoregressive block-based iterative encoder for efficient sequence modeling.
*arXiv preprint arXiv:2507.08567*, 2025.
- Bae et al. (2024)
Sangmin Bae, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Seungyeon Kim, and Tal Schuster.
Relaxed recursive transformers: Effective parameter sharing with layer-wise lora.
*arXiv preprint arXiv:2410.20672*, 2024.
- Bae et al. (2025)
Sangmin Bae, Yujin Kim, Reza Bayat, Sungnyun Kim, Jiyoun Ha, Tal Schuster, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Aaron Courville, et al.
Mixture-of-recursions: Learning dynamic recursive depths for adaptive token-level computation.
*arXiv preprint arXiv:2507.10524*, 2025.
- Biderman et al. (2023)
Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al.
Pythia: A suite for analyzing large language models across training and scaling.
In *International Conference on Machine Learning*, pp. 2397–2430. PMLR, 2023.
- Bisk et al. (2020)
Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al.
Piqa: Reasoning about physical commonsense in natural language.
In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pp. 7432–7439, 2020.
- Brown et al. (2020)
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.
Language models are few-shot learners.
*Advances in neural information processing systems*, 33:1877–1901, 2020.
- Chen et al. (2025a)
Xinghao Chen, Anhao Zhao, Heming Xia, Xuan Lu, Hanlin Wang, Yanjun Chen, Wei Zhang, Jian Wang, Wenjie Li, and Xiaoyu Shen.
Reasoning beyond language: A comprehensive survey on latent chain-of-thought reasoning.
*arXiv preprint arXiv:2505.16782*, 2025a.
- Chen et al. (2025b)
Yilong Chen, Junyuan Shang, Zhenyu Zhang, Yanxi Xie, Jiawei Sheng, Tingwen Liu, Shuohuan Wang, Yu Sun, Hua Wu, and Haifeng Wang.
Inner thinking transformer: Leveraging dynamic depth scaling to foster adaptive internal thinking.
*arXiv preprint arXiv:2502.13842*, 2025b.
- Chowdhery et al. (2023)
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al.
Palm: Scaling language modeling with pathways.
*Journal of Machine Learning Research*, 24(240):1–113, 2023.
- Clark et al. (2018)
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord.
Think you have solved question answering? try arc, the ai2 reasoning challenge.
*arXiv preprint arXiv:1803.05457*, 2018.
- Comanici et al. (2025)
Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al.
Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities.
*arXiv preprint arXiv:2507.06261*, 2025.
- Csordás et al. (2024)
Róbert Csordás, Kazuki Irie, Jürgen Schmidhuber, Christopher Potts, and Christopher D Manning.
Moeut: Mixture-of-experts universal transformers.
*Advances in Neural Information Processing Systems*, 37:28589–28614, 2024.
- Dao (2023)
Tri Dao.
Flashattention-2: Faster attention with better parallelism and work partitioning.
*arXiv preprint arXiv:2307.08691*, 2023.
- Dehghani et al. (2018)
Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser.
Universal transformers.
*arXiv preprint arXiv:1807.03819*, 2018.
- Elbayad et al. (2019)
Maha Elbayad, Jiatao Gu, Edouard Grave, and Michael Auli.
Depth-adaptive transformer.
*arXiv preprint arXiv:1910.10073*, 2019.
- Fan et al. (2024)
Ying Fan, Yilun Du, Kannan Ramchandran, and Kangwook Lee.
Looped transformers for length generalization.
*arXiv preprint arXiv:2409.15647*, 2024.
- Fedorenko et al. (2024)
Evelina Fedorenko, Steven T Piantadosi, and Edward AF Gibson.
Language is primarily a tool for communication rather than thought.
*Nature*, 630(8017):575–586, 2024.
- Gao et al. (2020)
Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al.
The pile: An 800gb dataset of diverse text for language modeling.
*arXiv preprint arXiv:2101.00027*, 2020.
- Gao et al. (2024)
Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou.
The language model evaluation harness, 07 2024.
URL [https://zenodo.org/records/12608602](https://zenodo.org/records/12608602).
- Geiping et al. (2025)
Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein.
Scaling up test-time compute with latent reasoning: A recurrent depth approach.
*arXiv preprint arXiv:2502.05171*, 2025.
- Giannou et al. (2023)
Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D Lee, and Dimitris Papailiopoulos.
Looped transformers as programmable computers.
In *International Conference on Machine Learning*, pp. 11398–11442. PMLR, 2023.
- Grattafiori et al. (2024)
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al.
The llama 3 herd of models.
*arXiv preprint arXiv:2407.21783*, 2024.
- Graves (2016)
Alex Graves.
Adaptive computation time for recurrent neural networks.
*arXiv preprint arXiv:1603.08983*, 2016.
- Hackenburg et al. (2025)
Kobi Hackenburg, Ben M Tappin, Paul Röttger, Scott A Hale, Jonathan Bright, and Helen Margetts.
Scaling language model size yields diminishing returns for single-message political persuasion.
*Proceedings of the National Academy of Sciences*, 122(10):e2413443122, 2025.
- Hao et al. (2024)
Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian.
Training large language models to reason in a continuous latent space.
*arXiv preprint arXiv:2412.06769*, 2024.
- Hay & Wolf (2024)
Tamir David Hay and Lior Wolf.
Dynamic layer tying for parameter-efficient transformers.
*arXiv preprint arXiv:2401.12819*, 2024.
- He et al. (2016)
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition.
In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.
- Hendrycks et al. (2020)
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
Measuring massive multitask language understanding.
*arXiv preprint arXiv:2009.03300*, 2020.
- Heo et al. (2025)
Jaemu Heo, Eldor Fozilov, Hyunmin Song, and Taehwan Kim.
Ringformer: Rethinking recurrent transformer with adaptive level signals.
*arXiv preprint arXiv:2502.13181*, 2025.
- Hoffmann et al. (2022)
Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al.
Training compute-optimal large language models.
*arXiv preprint arXiv:2203.15556*, 2022.
- Huang et al. (2017)
Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger.
Densely connected convolutional networks.
In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 4700–4708, 2017.
- Kaplan et al. (2020)
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.
Scaling laws for neural language models.
*arXiv preprint arXiv:2001.08361*, 2020.
- Kornblith et al. (2019)
Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton.
Similarity of neural network representations revisited.
In *International conference on machine learning*, pp. 3519–3529. PMlR, 2019.
- Li et al. (2025a)
Guanghao Li, Wenhao Jiang, Li Shen, Ming Tang, and Chun Yuan.
Zero token-driven deep thinking in llms: Unlocking the full potential of existing parameters via cyclic refinement.
*arXiv preprint arXiv:2502.12214*, 2025a.
- Li & Li (2021)
GuoLiang Li and Yiyang Li.
Recurrent multiple shared layers in depth for neural machine translation.
*arXiv preprint arXiv:2108.10417*, 2021.
- Li et al. (2025b)
Jindong Li, Yali Fu, Li Fan, Jiahong Liu, Yao Shu, Chengwei Qin, Menglin Yang, Irwin King, and Rex Ying.
Implicit reasoning in large language models: A comprehensive survey.
*arXiv preprint arXiv:2509.02350*, 2025b.
- Li et al. (2024)
Wenxue Li, Xiangzhou Liu, Yuxuan Li, Yilun Jin, Han Tian, Zhizhen Zhong, Guyue Liu, Ying Zhang, and Kai Chen.
Understanding communication characteristics of distributed training.
In *Proceedings of the 8th Asia-Pacific Workshop on Networking*, pp. 1–8, 2024.
- Liu et al. (2024)
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al.
Deepseek-v3 technical report.
*arXiv preprint arXiv:2412.19437*, 2024.
- Mohtashami et al. (2023)
Amirkeivan Mohtashami, Matteo Pagliardini, and Martin Jaggi.
Cotformer: A chain-of-thought driven architecture with budget-adaptive computation cost at inference.
*arXiv preprint arXiv:2310.10845*, 2023.
- Momeni et al. (2025)
Ali Momeni, Babak Rahmani, Benjamin Scellier, Logan G Wright, Peter L McMahon, Clara C Wanjura, Yuhang Li, Anas Skalli, Natalia G Berloff, Tatsuhiro Onodera, et al.
Training of physical neural networks.
*Nature*, 645(8079):53–61, 2025.
- Muennighoff et al. (2023)
Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin A Raffel.
Scaling data-constrained language models.
*Advances in Neural Information Processing Systems*, 36:50358–50376, 2023.
- Narayanan et al. (2021)
Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, et al.
Efficient large-scale language model training on gpu clusters using megatron-lm.
In *Proceedings of the international conference for high performance computing, networking, storage and analysis*, pp. 1–15, 2021.
- Ng & Wang (2024)
Kei-Sing Ng and Qingchen Wang.
Loop neural networks for parameter sharing.
*arXiv preprint arXiv:2409.14199*, 2024.
- Nguyen & Lin (2025)
Anthony Nguyen and Wenjun Lin.
Intra-layer recurrence in transformers for language modeling.
*arXiv preprint arXiv:2505.01855*, 2025.
- OpenAI (2023)
OpenAI.
Gpt-4 technical report: Tech. rep.
2023.
- Paperno et al. (2016)
Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández.
The lambada dataset: Word prediction requiring a broad discourse context.
*arXiv preprint arXiv:1606.06031*, 2016.
- Pati et al. (2023)
Suchita Pati, Shaizeen Aga, Mahzabeen Islam, Nuwan Jayasena, and Matthew D Sinclair.
Computation vs. communication scaling for future transformers on future hardware.
*arXiv preprint arXiv:2302.02825*, 2023.
- Patterson et al. (2021)
David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, and Jeff Dean.
Carbon emissions and large neural network training.
*arXiv preprint arXiv:2104.10350*, 2021.
- Pfau et al. (2024)
Jacob Pfau, William Merrill, and Samuel R Bowman.
Let’s think dot by dot: Hidden computation in transformer language models.
*arXiv preprint arXiv:2404.15758*, 2024.
- Sakaguchi et al. (2021)
Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi.
Winogrande: An adversarial winograd schema challenge at scale.
*Communications of the ACM*, 64(9):99–106, 2021.
- Saunshi et al. (2025)
Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J Reddi.
Reasoning with latent thoughts: On the power of looped transformers.
*arXiv preprint arXiv:2502.17416*, 2025.
- Shen et al. (2025)
Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He.
Codi: Compressing chain-of-thought into continuous space via self-distillation.
*arXiv preprint arXiv:2502.21074*, 2025.
- Snell et al. (2024)
Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar.
Scaling llm test-time compute optimally can be more effective than scaling model parameters.
*arXiv preprint arXiv:2408.03314*, 2024.
- Srivastava et al. (2015)
Rupesh Kumar Srivastava, Klaus Greff, and Jürgen Schmidhuber.
Highway networks.
*arXiv preprint arXiv:1505.00387*, 2015.
- Takase & Kiyono (2021)
Sho Takase and Shun Kiyono.
Lessons on parameter sharing across layers in transformers.
*arXiv preprint arXiv:2104.06022*, 2021.
- Tan et al. (2023)
Shawn Tan, Yikang Shen, Zhenfang Chen, Aaron Courville, and Chuang Gan.
Sparse universal transformer.
*arXiv preprint arXiv:2310.07096*, 2023.
- Vaswani et al. (2017)
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.
Attention is all you need.
*Advances in neural information processing systems*, 30, 2017.
- Villalobos et al. (2022)
Pablo Villalobos, Anson Ho, Jaime Sevilla, Tamay Besiroglu, Lennart Heim, and Marius Hobbhahn.
Will we run out of data? limits of llm scaling based on human-generated data.
*arXiv preprint arXiv:2211.04325*, 2022.
- Wei et al. (2022)
Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al.
Emergent abilities of large language models.
*arXiv preprint arXiv:2206.07682*, 2022.
- Welbl et al. (2017)
Johannes Welbl, Nelson F Liu, and Matt Gardner.
Crowdsourcing multiple choice science questions.
*arXiv preprint arXiv:1707.06209*, 2017.
- Xiao et al. (2025)
Da Xiao, Qingye Meng, Shengping Li, and Xingyuan Yuan.
Muddformer: Breaking residual bottlenecks in transformers via multiway dynamic dense connections.
*arXiv preprint arXiv:2502.12170*, 2025.
- Xu & Sato (2025)
Kevin Xu and Issei Sato.
To cot or to loop? a formal comparison between chain-of-thought and looped transformers.
*arXiv preprint arXiv:2505.19245*, 2025.
- Yang et al. (2023)
Liu Yang, Kangwook Lee, Robert Nowak, and Dimitris Papailiopoulos.
Looped transformers are better at learning learning algorithms.
*arXiv preprint arXiv:2311.12424*, 2023.
- Yu et al. (2025)
Qifan Yu, Zhenyu He, Sijie Li, Xun Zhou, Jun Zhang, Jingjing Xu, and Di He.
Enhancing auto-regressive chain-of-thought through loop-aligned reasoning.
*arXiv preprint arXiv:2502.08482*, 2025.
- Zellers et al. (2019)
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi.
Hellaswag: Can a machine really finish your sentence?
*arXiv preprint arXiv:1905.07830*, 2019.
- Zeng et al. (2025)
Boyi Zeng, Shixiang Song, Siyuan Huang, Yixuan Wang, He Li, Ziwei He, Xinbing Wang, Zhiyu Li, and Zhouhan Lin.
Pretraining language models to ponder in continuous space.
*arXiv preprint arXiv:2505.20674*, 2025.
- Zhang et al. (2024)
Xiang Zhang, Muhammad Abdul-Mageed, and Laks VS Lakshmanan.
Autoregressive+ chain of thought= recurrent: Recurrence’s role in language models’ computability and a revisit of recurrent transformer.
*arXiv preprint arXiv:2409.09239*, 2024.
- Zhu et al. (2024)
Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, and Xun Zhou.
Hyper-connections.
*arXiv preprint arXiv:2409.19606*, 2024.
- Zhu et al. (2025a)
Rui-Jie Zhu, Tianhao Peng, Tianhao Cheng, Xingwei Qu, Jinfa Huang, Dawei Zhu, Hao Wang, Kaiwen Xue, Xuanliang Zhang, Yong Shan, et al.
A survey on latent reasoning.
*arXiv preprint arXiv:2507.06203*, 2025a.
- Zhu et al. (2025b)
Ruike Zhu, Hanwen Zhang, Tianyu Shi, Chi Wang, Tianyi Zhou, and Zengyi Qin.
The 4th dimension for scaling model size.
*arXiv preprint arXiv:2506.18233*, 2025b.

## Appendix A Related work

Recursive Transformers and Loop-based LLMs. The idea of iterating a Transformer layer in a loop originates from the Universal Transformer (UT) (Dehghani et al., 2018), which showed that repeatedly applying a single, weight-shared layer can achieve the expressive power of a much deeper Transformer while allowing variable computation per input. UT also introduced an Adaptive Computation Time mechanism (Graves, 2016) that dynamically adjusts how many iterations to run for each token, together with a trainable positional signal that distinguishes time steps. Since UT, a series of works have extended the concept of looped Transformers (Tan et al., 2023; Giannou et al., 2023; Li & Li, 2021; Takase & Kiyono, 2021; Elbayad et al., 2019; Yang et al., 2023; Zhang et al., 2024; Fan et al., 2024; Hay & Wolf, 2024; Chen et al., 2025b; Nguyen & Lin, 2025; Li et al., 2025a; Aleksandrov et al., 2025; Bae et al., 2025). Recent studies (Saunshi et al., 2025; Zhu et al., 2025b; Geiping et al., 2025) demonstrated—both empirically and theoretically—that increasing depth by looping a small Transformer can match or surpass a far deeper fixed-depth model on challenging reasoning tasks. Later, Zeng et al. (2025) reinforced the view that iterative “pondering” could be critical for test-time scaling and linked the behavior of looped Transformers to an implicit chain-of-thought process. Collectively, these efforts underscore the promise of recursive transformers for adaptive depth and latent reasoning.

Parameter Sharing and Iteration Differentiation. A central challenge for recursive transformers is to preserve expressiveness even though all iterations reuse the same parameters. An empirical study of parameter sharing in Transformers showed that naïvely sharing every layer—as in the original UT (Dehghani et al., 2018) —often degrades performance on language tasks, implying that additional mechanisms are required to alleviate representational bottlenecks (Ng & Wang, 2024). Several strategies have been explored: Learned loop-index embeddings. By injecting a small trainable vector or matrix that encodes the iteration number, models can behave slightly differently at each step while still sharing the main weights (Dehghani et al., 2018; Mohtashami et al., 2023). However, element-wise addition of such embeddings practically produces limited gains (Geiping et al., 2025; Zhu et al., 2025a). LoRA per iteration. In a similar spirit, recent works (Heo et al., 2025; Bae et al., 2024) attach a separate low-rank adaptation (LoRA) module to each repetition of a pre-trained model, granting every loop its own lightweight set of parameters and mitigating the drawbacks of strict sharing. Mixture-of-Experts in a loop. MoEUT (Csordás et al., 2024) combines weight sharing with a Mixture-of-Experts (MoE) at every layer: the base layer is reused across iterations, while expert gating adds conditional capacity. MoEUT slightly outperforms a non-loop Transformer of equal compute, underscoring the value of learnable gating and expert routes within a loop architecture.
Our work proposes a different paradigm. Instead of adding unique parameters to the loop core to differentiate iterations, we focus on dynamically managing the information flow itself. We propose MeSH to externalize state management into a memory buffer and employ lightweight, step-wise routers to control what is read from and written to it. This reframes the problem of iteration differentiation from one of adding capacity to one of explicit, learnable routing.

Skip Connections and Dense Connectivity. In deep networks, skip or shortcut connections have long been essential for training very deep architectures effectively (He et al., 2016). Residual Networks (ResNets) and Highway Networks showed that adding identity skip paths improves gradient flow and allows each layer to learn a simpler “update function” on top of an identity mapping He et al. (2016); Srivastava et al. (2015). DenseNet (Huang et al., 2017) further generalized this idea by connecting every layer to all previous layers, so that each layer receives the feature maps of all preceding layers as input; this dense feed-forward architecture promotes feature reuse, mitigates vanishing gradients, and even reduces parameter count. More recently, Hyper-connections (Zhu et al., 2024) widen the hidden state into multiple parallel streams and use learnable coefficients to mix these streams, effectively replacing the standard residual path with a more complex, multi-lane data highway. MUDDformer (Xiao et al., 2025) introduced dense connections into standard decoder-only Transformers, generalizing residuals by adding multiple learnable skip paths between layers; dynamic dense skips allowed a 2.8 B model to match the perplexity of a 6.9 B model with only minimal overhead (Xiao et al., 2025).
While conceptually related to dense connectivity, our work focuses on block-to-block (loop-to-loop) connectivity in the recursive setting rather than layer-to-layer wiring.
We diverge from direct connections by introducing an external memory buffer and lightweight, step-wise routers. The system facilitates a flexible read-write cycle for managing information flow across iterations, rather than simply gating feed-forward paths. The mechanism is both principled and more native to the recursive design, as it is explicitly engineered to enable functional specialization between iterations.

Latent Reasoning and Chain-of-Thought. Loop-based LLMs are closely related to the idea of latent (hidden) chain-of-thought (Hao et al., 2024; Shen et al., 2025). Instead of explicitly outputting intermediate reasoning steps in natural language, a loop-based model processes those steps internally in vector form (Zhu et al., 2025a; Xu & Sato, 2025). Recent research has examined the differences between prompting a model with an explicit chain-of-thought versus giving it the capacity to “think” silently via latent reasoning (Chen et al., 2025a; Fedorenko et al., 2024; Hao et al., 2024; Pfau et al., 2024). In general, loop offers a promising way to achieve the benefits of multi-step reasoning without incurring the cost of longer outputs or the need for supervised intermediate steps (Li et al., 2025b). Our work contributes to this area by improving the recursive architectural foundation on which latent reasoning unfolds. By enhancing how information is preserved and combined across iterative steps, we aim to make iterative execution more effective. While approaches like CoTFormer inject special tokens to mimic multi-step reasoning inside the model (Mohtashami et al., 2023), most of recursive transformers focus on intrinsic connectivity of loop iterations (Zeng et al., 2025; Geiping et al., 2025; Saunshi et al., 2025). These two methodologies can be seen as complementary, as we can imagine a loop-based LLM that also uses latent CoT training signals. Indeed, analysis of hidden state evolution under different connection schemes can be viewed as an interpretability study of latent reasoning – shedding light on whether the model is gradually refining a solution or oscillating, and how much it relies on initial information versus newly computed results at each step.

## Appendix B Experimental Details

#### Pre-training.

All models are pretrained from scratch, closely following the methodology of the Pythia suite (Biderman et al., 2023). Our training is conducted on a 250B-token deduplicated subset of the Pile dataset (Gao et al., 2020), using the original GPT-NeoX tokenizer with a vocabulary size of 50,257. All models are trained for one epoch.

#### Model Architecture.

Our implementations are based on the GPT-NeoX architecture provided by the Pythia suite (Biderman et al., 2023). For recursive models, we adopt the prelude-loop-coda structure.
We denote the layer distribution using the notation Lpre+LcoreRNloop+Lcoda. For example, a 4+8R2+4 configuration corresponds to a model with a 4-layer prelude (Lpre), an 8-layer shared core (Lcore) that is looped twice (Nloop), and a 4-layer coda (Lcoda).
Our +mesh variant is implemented by inserting a state buffer and step-wise routers at the boundaries of these conceptual blocks. Each router consists of a single linear layer followed by a softmax function to generate dynamic routing weights. The buffer size $B$ is set following the empirical $B=N_{\text{loop}}+3$ derived from our ablation study (see Appendix [E.1](https://arxiv.org/html/2510.07739v1#A5.SS1)). Following standard Transformer practice (Vaswani et al., 2017), we scale the input embeddings by a factor of $\sqrt{d_{\text{model}}}$ before they enter the first layer. To ensure training stability in these deep computational graphs, we employ a depth-aware weight initialization, scaling the standard deviation of output projection weights by $1/\sqrt{2\times N_{\text{compute}}}$, where $N_{\text{compute}}$ is the total number of layers in the unrolled computation graph.

#### Training Hyperparameters.

We use the AdamW optimizer with $\beta_{1}=0.9$, $\beta_{2}=0.95$, and a weight decay of $0.01$. The learning rate follows a cosine decay schedule with a 1% warmup, decaying to 10% of the peak value. The peak learning rate is scaled according to model size, ranging from $6.0\times 10^{-4}$ for the 160M model to $2.0\times 10^{-4}$ for the 1.4B model. All models are trained with a consistent global batch size of 512 and a sequence length of 4096 tokens. To improve training efficiency, we utilize BF16 mixed-precision and FlashAttention-2 (Dao, 2023). Our distributed training setup is managed by DeepSpeed with ZeRO Stage 0.

#### Downstream Task Evaluation.

To assess model performance, we evaluated few-shot accuracy on 9 benchmarks using the Language Model Evaluation Harness framework (Gao et al., 2024). The evaluation suite includes: Lambada (Paperno et al., 2016) in both its OpenAI (LD-O) and Standard (LD-S) versions, PIQA (PQ) (Bisk et al., 2020), HellaSwag (HS) (Zellers et al., 2019), WinoGrande (WG) (Sakaguchi et al., 2021), ARC-Easy (ARC-E) and ARC-Challenge (ARC-C) (Clark et al., 2018), SciQ (Welbl et al., 2017), and continuation-MMLU (cMMLU)  (Hendrycks et al., 2020). We report accuracy normalized by the byte length of the target string for PIQA, HellaSwag, ARC-E, ARC-C, and SciQ and standard accuracy for Lambada, WinoGrande, and cMMLU. All evaluations are conducted in both 0-shot and 5-shot settings. All measurements were performed on a single NVIDIA H20 GPU. Detailed results are shown in Table [2](https://arxiv.org/html/2510.07739v1#A2.T2).

Table: Table 2: Detailed downstream evaluation results (stacked). For each model variant, performance is shown for both 0-shot and 5-shot settings. We report accuracy values for all tasks. The average accuracy (“Avg.”) is computed over the 9 preceding tasks. Dataset abbreviations correspond to: LD-O (Lambada OpenAI), LD-S (Lambada Standard), HS (HellaSwag), PQ (PIQA), WG (WinoGrande), ARC-E (ARC-easy), ARC-C (ARC-Challenge), SciQ (SciQ), and cMMLU (MMLU-continuation).

## Appendix C Pseudocode

We provide detailed pseudocode for the recursive architectures discussed in the main paper. Algorithm [1](https://arxiv.org/html/2510.07739v1#alg1) outlines the implementation of common recursive variants, which rely on fixed, heuristic-based state-passing schemes. In contrast, Algorithm [2](https://arxiv.org/html/2510.07739v1#alg2) details our proposed MeSH-augmented recurrence, which replaces the rigid logic with a dynamic, memory-based system.

Figure: Algorithm 1 Recursive Transformers with Common Variants

Figure: Algorithm 2 MeSH-Augmented Recurrence within a Prelude-Recurrent-Coda Structure

## Appendix D Discussion: Expressive Power of MeSH as a General Recurrence

The baseline recurrences described in Section [3.1](https://arxiv.org/html/2510.07739v1#S3.SS1) employ a fixed, non-adaptive state update rule: the output of the core block, $\mathbf{h}_{\text{m}}^{(t)}=f_{\text{core}}(\mathbf{h}^{(t)})$, is always supplemented by a predetermined state (e.g., zero, the previous state $\mathbf{h}^{(t)}$, or the initial state $\mathbf{h}^{(0)}$). We propose that MeSH offers a more general and powerful alternative by replacing this rigid addition with a learnable, dynamic state composition mechanism.

#### Proposition 2.1.

The MeSH recurrence, defined by the compute-write-read cycle, generalizes the concept of additive state updates (as in residual and anchor variants) by learning to dynamically retrieve and combine historical states from memory to form the state for the next iteration.

#### Demonstration.

To reveal the underlying mechanics, we can unroll the MeSH update equations. The state for the next iteration, $\mathbf{h}^{(t+1)}$, is formed by reading from the just-updated memory $\mathbf{M}^{(t+1)}$:

$$ $\displaystyle\mathbf{h}^{(t+1)}$ $\displaystyle=\sum_{b=0}^{B-1}\mathbf{m}_{b}^{(t+1)}\odot\mathbf{w}_{\text{read},b}^{(t)}$ $\displaystyle=\sum_{b=0}^{B-1}\left(\mathbf{m}_{b}^{(t)}+\mathbf{h}_{\text{m}}^{(t)}\odot\mathbf{w}_{\text{write},b}^{(t)}\right)\odot\mathbf{w}_{\text{read},b}^{(t)}$ $\displaystyle=\underbrace{\sum_{b=0}^{B-1}\mathbf{m}_{b}^{(t)}\odot\mathbf{w}_{\text{read},b}^{(t)}}_{\text{Retrieved Historical Summary}}+\underbrace{\left(\sum_{b=0}^{B-1}\mathbf{w}_{\text{write},b}^{(t)}\odot\mathbf{w}_{\text{read},b}^{(t)}\right)}_{\text{Gating Factor}}\odot\mathbf{h}_{\text{m}}^{(t)}$ (6) $$

Let us analyze the two resulting components. The second term is the core’s output, $\mathbf{h}_{\text{m}}^{(t)}$, scaled by a learned gating factor. The first term is a dynamic retrieval of information from the memory state $\mathbf{m}^{(t)}$ before the current write operation. Note that this term is distinct from the previous state $\mathbf{h}^{(t)}$, which was formed using the read weights from the prior step, $\mathbf{w}_{\text{read}}^{(t-1)}$.

The formulation reveals that the next state $\mathbf{h}^{(t+1)}$ is a generalized residual update composed of:

- 1.
A retrieved historical summary that dynamically combines states presented in the buffer. The read router learns what historical information is most relevant at this step.
- 2.
A gated output of the current core block, $\mathbf{h}_{\text{m}}^{(t)}$, scaled by a learned gating factor.

The dynamic process generalizes the fixed baseline recurrences, as we can conceptually unroll the retrieved historical summary even further as a weighted combination of the initial memory state and all previous core outputs $\{\mathbf{h}_{\text{m}}^{(0)},\dots,\mathbf{h}_{\text{m}}^{(t-1)}\}$ that have been written to the buffer. Therefore, the next state $\mathbf{h}^{(t+1)}$ can be viewed as a comprehensive, dynamic aggregation of all computations performed so far:

$$ $\mathbf{h}^{(t+1)}=\alpha_{t}\odot\mathbf{h}_{\text{m}}^{(t)}+\sum_{i=0}^{t-1}\alpha_{i}\odot\mathbf{h}_{\text{m}}^{(i)}+\alpha_{\text{emb}}\odot\mathbf{h}_{\text{emb}}$ (7) $$

where all coefficients $\alpha$ are implicit coupled with write and read weights during previous iterations. The perspective makes the generalization self-evident:

- •
Simulating anchor ($\mathbf{h}_{\text{m}}^{(t)}+\mathbf{h}_{\text{emb}}$):
This is achieved by learning a routing scheme where the coefficients in Eq. [7](https://arxiv.org/html/2510.07739v1#A4.E7) are set as follows: the weight for the current computation, $\alpha_{t}$, approaches 1; the weight for the initial embedding, $\alpha_{\text{emb}}$, approaches 1; and all other historical weights, $\alpha_{i<t}$, are driven to zero. MeSH can learn to adopt this specific weighting only when needed, rather than being hard-wired to it.
- •
Simulating residual ($\mathbf{h}_{\text{m}}^{(t)}+\mathbf{h}^{(t)}$):
To approximate this, MeSH needs to reconstruct the previous state, $\mathbf{h}^{(t)}$, as its historical summary. This is naturally achievable. Since $\mathbf{h}^{(t)}$ is itself a weighted sum of $\{\mathbf{h}_{\text{emb}},\mathbf{h}_{\text{m}}^{(0)},\dots,\mathbf{h}_{\text{m}}^{(t-1)}\}$, the routers at step $t$ can learn to compute the appropriate coefficients ($\alpha_{\text{emb}},\alpha_{0},\dots,\alpha_{t-1}$) to reconstruct or closely approximate $\mathbf{h}^{(t)}$. More powerfully, MeSH can choose to form a “better” historical summary by up-weighting more relevant past states (e.g., $\mathbf{h}_{\text{m}}^{(t-5)}$) and down-weighting irrelevant ones (e.g., $\mathbf{h}_{\text{m}}^{(t-1)}$), thus forming more effective long-range dependencies.
- •
Adaptive Combination: The core advantage is that the coefficients $\alpha$ are not fixed. They are functions of the current state $\mathbf{h}^{(t)}$, allowing the model to change its recurrence rule on the fly. It can learn to behave like an Anchor in early steps, transition to a Residual-like update, and synthesize a complex summary from multiple past states for the final output, all within a single forward pass.

In conclusion, MeSH does not just replicate the fixed recurrences; it subsumes the underlying principle of combining past and present information into a flexible, learnable framework. It replaces the hard-coded “what to add” (e.g., $\mathbf{h}^{(0)}$ or $\mathbf{h}^{(t)}$) with a learned “what to retrieve and combine,” offering a substantially more expressive mechanism for managing state in recursive transformers.

## Appendix E More Results

### E.1 Ablation Study: MeSH Buffer Length

**Table 3: Ablation on MeSH buffer length for the Pythia-410M model ($N_{\text{loop}}=2$). Performance is optimized with $k=2$ auxiliary “scratchpad” slots.**
| Scratchpad Slots ($k$) | Buffer Length ($B=(N_{\text{loop}}+1)+k$) | Loss $\downarrow$ | PPL $\downarrow$ |
| --- | --- | --- | --- |
| 0 | $(2+1)+0=3$ | 2.2108 | 9.1231 |
| 1 | $(2+1)+1=4$ | 2.2083 | 9.1003 |
| \cellcolormeshgreen2 | \cellcolormeshgreen$(2+1)+2=5$ | \cellcolormeshgreen2.2077 | \cellcolormeshgreen9.0944 |
| 3 | $(2+1)+3=6$ | 2.2092 | 9.1088 |

To establish a principled heuristic for setting the MeSH buffer length ($B$), we hypothesize that its capacity should scale with the number of major computational states generated during the recursive process. For a model with $N_{\text{loop}}$ iterations, this includes the initial state from the prelude network plus the output from each of the $N_{\text{loop}}$ core blocks, totaling $N_{\text{loop}}+1$ essential states. We therefore model the buffer size as $B=(N_{\text{loop}}+1)+k$, where $k$ is the number of auxiliary “scratchpad” slots available for flexible composition.

We conduct an ablation study to find the optimal $k$ using the Pythia-410M model with a 4+8R2+4 configuration, where $N_{\text{loop}}=2$. The results are shown in Table [3](https://arxiv.org/html/2510.07739v1#A5.T3). Performance improves as we add scratchpad slots, peaking at $k=2$. This configuration, corresponding to a total buffer length of $B=5$, achieves the lowest validation loss and perplexity. Performance slightly degrades at $k=3$, suggesting a point of diminishing returns. This indicates a sweet spot where the buffer has dedicated slots for each major computational state, plus two auxiliary slots for managing intermediate representations, without making the routing task overly complex. Based on empirical results, we adopt the general rule $B=N_{\text{loop}}+3$ for all MeSH models in our main experiments.

### E.2 Ablation Study: Heuristic State-Passing Schemes

While individual heuristic schemes like +anchor and +residual improve over the base recursive model, a natural question arises: can we achieve further gains by combining them, and can the model learn the optimal combination? To investigate this, we conducted an ablation study on the Pythia-410M model (4+8R2+4 configuration) exploring both fixed additive combinations and learnable linear combinations of supplementary states ($\mathbf{h}_{\text{sup}}^{(t)}$).
For the learnable schemes, we define the supplementary state as a combination of the base, anchor, anchor*, and residual states:
$\mathbf{h}^{(t)}=\alpha_{1}\mathbf{h}_{\text{m}}^{(t)}+\alpha_{2}\mathbf{h}^{(0)}+\alpha_{3}\mathbf{h}_{\text{emb}}$. We test two variants for the coefficients $\alpha_{i}$:

- •
Static Combination: The coefficients are trainable scalar parameters that are fixed after training (Ng & Wang, 2024).
- •
Dynamic Combination: The coefficients are dynamically computed at each iteration based on the previous state $\mathbf{h}^{(t-1)}$.

The results, summarized in Table [4](https://arxiv.org/html/2510.07739v1#A5.T4), reveal the brittleness of heuristic design. Simply adding all states (+residual+anchor+anchor*) degrades performance, yielding a higher perplexity (9.24) than using +anchor alone (9.19). This confirms that a naive “more is better” approach is not a reliable strategy. While a carefully hand-picked combination (+anchor+anchor*) achieves the best result among all explicit schemes (9.17 PPL), this requires manual tuning. The learnable static and dynamic combinations effectively avoid the worst-case performance degradation but fail to match the best-performing heuristic.

**Table 4: Ablation on combinations of supplementary context schemes for Pythia-410M (4+8R2+4). We report evaluation loss and perplexity on the Pile dataset. While a hand-picked combination (+anchor+anchor*) works best among heuristics, MeSH surpasses all explicit schemes.**
| Scheme | Loss $\downarrow$ | PPL $\downarrow$ |
| --- | --- | --- |
| \cellcolorVanillaCvanilla | \cellcolorVanillaC2.2047 | \cellcolorVanillaC9.0675 |
| \cellcolorBaseCrecursive-base | \cellcolorBaseC2.2358 | \cellcolorBaseC9.3542 |
| Single Heuristic Baselines |  |  |
| +anchor | 2.2178 | 9.1867 |
| Fixed Additive Combinations |  |  |
| +residual+anchor | 2.2251 | 9.2541 |
| +residual+anchor+anchor* | 2.2237 | 9.2415 |
| +anchor+anchor* | 2.2159 | 9.1694 |
| Learnable Combinations |  |  |
| +static comb. | 2.2163 | 9.1731 |
| +dynamic comb. | 2.2176 | 9.1851 |
| \cellcolormeshgreen+mesh (ours) | \cellcolormeshgreen2.2077 | \cellcolormeshgreen9.0944 |

This suggests that while explicit, learnable weighting can provide a “safe” baseline by ignoring detrimental combinations, it lacks the expressive capacity to discover optimal synergistic interactions. In sharp contrast, our +mesh model (9.09 PPL) significantly outperforms all heuristic-based approaches. Instead of being constrained to an explicit, low-dimensional linear combination of predefined states, MeSH learns a complex, non-linear function for retrieving and composing information from its memory buffer. This allows it to discover implicit, high-dimensional combinations, effectively breaking through the performance ceiling imposed by simpler, explicit state-passing schemes.

### E.3 Complexity Analysis

**Table 5: Parameter counts for Pythia-1.4B variants. Percentages show the change relative to the vanilla baseline for total and non-embedding parameters.**
| Model | Variant | Config | Total Params | Non-Embedding<br>Params | Router<br>Weights |
| --- | --- | --- | --- | --- | --- |
| Pythia-1.4B | vanilla | 24 | 1,423,036,416 | 1,208,602,624 | / |
| recursive | 4+8R2+4 | 1,020,170,240<br>(-28.310%) | 805,736,448<br>(-33.333%) | / |  |
| \cellcolormeshgreen+mesh | \cellcolormeshgreen<br>4+8R2+4<br>($B=5$) | \cellcolormeshgreen<br>1,020,231,710<br>(-28.306%) | \cellcolormeshgreen<br>805,797,918<br>(-33.328%) | \cellcolormeshgreen<br>61,470<br>(+0.005%) |  |

MeSH introduces a set of lightweight, step-wise routers for its read and write operations. The total number of additional parameters is determined by $(N_{\text{loop}}+1)\times D_{\text{hidden}}\times B\times 2$, where $N_{\text{loop}}$ is the number of loop iterations, $D_{\text{hidden}}$ is the hidden size, $B$ is the buffer length, and the factor of 2 accounts for both read and write routers. This overhead is negligible compared to the significant parameter savings achieved through recursion. As detailed in Table [5](https://arxiv.org/html/2510.07739v1#A5.T5), for our Pythia-1.4B model in a 4+8R2+4 configuration, the recursive structure reduces the non-embedding parameters by 33.33% compared to its vanilla counterpart. The MeSH routers add a mere 61,470 parameters (0.005% relative to the non-embedding part), which shows that MeSH achieves its substantial performance gains with virtually no cost to parameter efficiency, making it an architecturally lightweight yet powerful enhancement.

### E.4 Detailed Training Dynamics on Downstream Tasks

To provide a more granular view of the training dynamics presented in Section [4.3](https://arxiv.org/html/2510.07739v1#S4.SS3), Figure [9](https://arxiv.org/html/2510.07739v1#A5.F9) shows the performance of the 1.4B-parameter models on 9 individual downstream tasks and their average accuracy, evaluated at various checkpoints throughout the pre-training process.

Figure: Figure 9: Detailed Training Dynamics of 1.4B Recursive Variants on Downstream Tasks. Each panel displays the 0-shot and 5-shot accuracy for one of the 9 individual downstream tasks or their overall average (“Avg. Accuracy”), evaluated at different checkpoints throughout the 120,818-step pre-training process.
Refer to caption: x9.png

### E.5 Applying MeSH to Non-Recursive Architectures

In our main experiments, the recursive+mesh model for Pythia-1.4B surpasses its larger Vanilla counterpart, even with 33.3% fewer parameters. The result suggests that the MeSH mechanism might offer architectural benefits beyond the the recursive setting. We hypothesize that if the performance bottleneck of parameter sharing were removed, the benefits of MeSH could be even more pronounced. We conduct an experiment applying a MeSH-like structure to a standard, non-recursive Vanilla transformer. We conceptually partition the 24 layers of the Pythia-1.4B Vanilla model into blocks that mirror our 4+8R2+4 recursive design: a 4-layer prelude, two distinct 8-layer core blocks (core_1 and core_2), and a 4-layer coda. Crucially, unlike in the recursive setup, core_1 and core_2 do not share weights. The MeSH mechanism, with its memory buffer and routers, is then inserted at the boundaries between these conceptual blocks to manage information flow. Results are shown in Table [6](https://arxiv.org/html/2510.07739v1#A5.T6). The vanilla+mesh model achieves a lower perplexity (7.26) than the standard Vanilla baseline (7.44), confirming that MeSH provides a direct performance uplift even without the constraint of parameter sharing. This finding provides a compelling explanation for the strong performance of our main recursive+mesh model: the architectural benefits of MeSH are potent enough to not only compensate for the performance loss typically incurred by weight sharing but to exceed the original baseline.

**Table 6: Performance comparison of applying MeSH to recursive and non-recursive (Vanilla) backbones on the Pythia-1.4B scale. All metrics are evaluated on the Pile dataset.**
| Variant | Config | Non-Emb<br>Params (%) | Loss $\downarrow$ | PPL $\downarrow$ |
| --- | --- | --- | --- | --- |
| vanilla | 24 layers | 100% | 2.0070 | 7.4406 |
| \cellcolormeshgreenvanilla+mesh | \cellcolormeshgreen4+8+8+4 | \cellcolormeshgreen100% | \cellcolormeshgreen1.9818 | \cellcolormeshgreen7.2559 |
| recursive (base) | 4+8R2+4 | 66.7% | 2.0317 | 7.6267 |
| \cellcolormeshgreenrecursive+mesh | \cellcolormeshgreen4+8R2+4 | \cellcolormeshgreen66.7% | \cellcolormeshgreen1.9996 | \cellcolormeshgreen7.3865 |

While our MeSH framework was considered in recursive transformers, the result indicates that its core principle of explicit, routed state management has broader applicability. Exploring MeSH as a general architectural primitive for enhancing deep, non-recursive transformers is a promising direction for our future research.

## Appendix F Limitations and Future Work

While this work establishes MeSH as a promising architectural principle for recursive transformers, we recognize several limitations that open up avenues for future research. First, our experiments have validated the effectiveness of MeSH on models up to the Pythia-1.4B scale, trained on the deduplicated Pile dataset. It remains unclear whether the parameter-efficiency gains persist at larger scales and under different training regimes. A natural and important direction for future work is to apply and evaluate the MeSH architecture on state-of-the-art foundation models at much larger scales. Second, although our ablation study reveals that MeSH can also benefit non-recursive transformers, a comprehensive investigation beyond recursive backbones falls outside the scope of this paper. Exploring MeSH as a general-purpose architectural primitive for improving information flow remains a promising direction for our future work.
