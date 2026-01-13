# HoloGraph V7 Architecture

HoloGraph V7 is a hybrid sequence model that combines **State Space Models (SSM)**, **Box Embeddings**, and an **External Relational Memory** system. Unlike standard Transformers, it processes long-range dependencies through a selective holographic scan and a persistent archive.

## Core Components

### 1. Box Embeddings
Instead of representing tokens as single points in vector space, HoloGraph uses **Box Embeddings**. Each token is defined by a **Center** and an **Offset**, creating an $n$-dimensional hyper-rectangle (interval). This allows the model to represent hierarchy and containment relationships mathematically through box intersections.

### 2. Selective Holographic Scan (SSM)
The temporal engine is a custom **Parallel Scan** mechanism. 
- It uses gating units (`gate_write` and `gate_forget`) to control information flow.
- It operates with **Log-space stability** to prevent gradient explosions.
- It calculates a cumulative state across the sequence, similar to Modern State Space Models (Mamba), but optimized for holographic state updates.

### 3. Relational Holo Archive
The model maintains an external memory bank (the **Archive**) that persists across training steps:
- **Snapshots:** Periodically, the model takes a "snapshot" of its internal state and box boundaries.
- **Relational Retrieval:** During the forward pass, the model queries the archive. It calculates similarity based on the **log-volume overlap** between the current input boxes and the archived boxes.
- **Motif Injection:** Retrieved information is fused back into the layers as a "motif context," allowing the model to recall patterns from long-past sequences.

### 4. Layer Structure
Each **HoloGraphBlock** consists of:
- **Normalization:** LayerNorm pre-processing.
- **Projection:** Linear projections for Keys, Queries, Values, and Gating parameters.
- **Selective Scan:** The primary sequence processing bottleneck.
- **FFN:** A standard Gated Linear Unit (GELU) feed-forward network.
- **Residual Connections:** Additive paths for both the scan output and the retrieved archive context.

## Technical Specifications
- **Vocab Representation:** Box-bounded intervals.
- **Recurrence:** Log-stable parallel cumulative sum.
- **Memory:** Cross-batch relational archive with intersection-based retrieval.
- **Optimization:** Supports 8-bit AdamW and Gradient Checkpointing for memory efficiency.
