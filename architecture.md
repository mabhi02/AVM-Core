# AVM-CORE Architecture

This document describes the architecture and training pipeline for the Adaptive Validation Model with Compositional Reasoning (AVM-CORE), a deep learning system designed for mathematical proof generation and validation.

## Data Architecture

### Data Schema

Training data consists of JSON files with the following structure:

```json
{
  "theorem": "String containing the mathematical theorem",
  "symbols": ["List", "of", "mathematical", "symbols"],
  "steps": ["Step 1 of proof", "Step 2 of proof", "..."],
  "relations": [{"from_idx": 0, "to_idx": 1, "type_idx": 0}, ...],
  "strategy_labels": [0, 2, 1, ...],
  "difficulty": 0.75
}
```

### Data Processing Pipeline

- **Initial Processing**: `avm/data/processors/proof_processor.py` converts raw proofs into structured format
- **Symbol Processing**: `avm/data/processors/symbol_processor.py` handles mathematical symbol extraction
- **Dataset Creation**: `avm/data/dataset.py` implements the PyTorch Dataset for loading and processing proofs

### Tokenization and Embedding

- **Tokenizer** (`avm/utils/tokenizer.py`):
  - Mathematical symbol tokenization
  - Special tokens: `<PAD>`, `<UNK>`, `<START>`, `<END>`
  - Regular expression patterns for mathematical symbols

- **Embeddings** (`avm/training/embeddings.py`):
  - Token embeddings
  - Positional encodings
  - Type embeddings (theorem, proof steps, generated content)

## Model Architecture

### Core Model Components

- **Main Model Class**: `AVMCore` in `avm/models/proof_generator.py`

- **Model Dimensions**:
  - Default configuration: 256-dim hidden size, 8 attention heads, 6 layers (~20-30M parameters)
  - Memory-efficient configuration: 128-dim hidden size, 4 attention heads, 4 layers (~5-10M parameters)

- **Dual Decoder Architecture**:
  - Understanding decoder: Processes input theorems and existing proof steps
  - Generation decoder: Generates new proof steps with causal attention masking

- **Strategy Router**:
  - Component that selects between proof strategies
  - Implemented in `StrategyRouter` class within `avm/models/proof_generator.py`

### Reasoning Components

- **Compositional Engine** (`avm/core/compositional.py`):
  - Integrates graph reasoning with transformer-based sequence processing
  - Core component for mathematical concept reasoning

- **Graph Reasoning Engine** (`avm/core/graph_engine.py`):
  - Graph neural network for mathematical concept representation
  - Uses GATConv layers for node interactions

- **Validation Engine** (`avm/core/validator.py`):
  - Verifies proof steps and logical consistency
  - Produces validation scores at step and global levels

### Proof Strategies

Implemented in `avm/models/strategies.py`:

- Direct proof strategy
- Contradiction strategy
- Induction strategy
- Construction strategy
- Reduction strategy

## Training Pipeline

### Training Configuration

- **Main Training Scripts**:
  - `main.py`: Entry point with various modes
  - `avm/training/train.py`: Detailed training implementation
  - `avm/training/trainer.py`: Memory-efficient training wrapper

- **Configuration Files**:
  - `configs/model_config.yaml`: Model architecture parameters
  - `configs/training_config.yaml`: Training hyperparameters

### Training Parameters

- **Basic Parameters**:
  - Batch size: 2 (low for memory efficiency)
  - Gradient accumulation steps: 8 (effective batch size of 16)
  - Learning rate: 1e-4
  - Maximum epochs: 100
  - Validation interval: Every epoch
  - Early stopping patience: 10 epochs

- **Optimization**:
  - Optimizer: AdamW with weight decay of 0.01
  - Learning rate scheduler: Cosine annealing to 1e-6
  - Gradient clipping: 1.0

### Memory Optimization Techniques

- Mixed precision training (FP16)
- Gradient accumulation
- Chunked sequence processing
- CUDA memory management
- Reduced model dimensions when needed

### Training Process

1. **Data Loading and Preparation**:
   - Load processed proof data
   - Generate batches with proper padding and collation
   - Handle variable-length sequences

2. **Training Loop**:
   - Forward pass with automatic mixed precision
   - Loss calculation (generation loss + strategy selection loss)
   - Backward pass with gradient scaling
   - Weight updates after accumulation steps
   - Validation after each epoch
   - Checkpoint saving
   - Learning rate updates
   - Early stopping check

3. **Loss Calculation**:
   - Generation loss: Cross-entropy for next-token prediction
   - Strategy loss: Classification loss for strategy selection
   - Total loss: generation_loss + 0.5 * strategy_loss

## Inference Process

- **Beam Search Generation**:
  - Implemented in `AVMCore.beam_search_generate`
  - Default beam size: 2-5 (configurable)
  - Maximum sequence length configurable

- **Proof Generation Script**:
  - `scripts/generate_proof.py`
  - Takes a theorem as input
  - Outputs generated proof and visualization

- **Evaluation Metrics**:
  - Step accuracy
  - Strategy accuracy
  - Proof validity
  - Completeness

## Hardware Requirements

- GPU with at least 8GB VRAM for memory-efficient configuration
- 16GB+ VRAM recommended for full model
- Multi-day training process depending on dataset size and hardware

## File Structure Overview

```
avm/
├── core/
│   ├── compositional.py      # Compositional reasoning engine
│   ├── graph_engine.py       # Graph neural network component
│   └── validator.py          # Proof validation component
├── data/
│   ├── dataset.py            # PyTorch dataset implementation
│   ├── processors/           # Data processing utilities
│   └── scrapers/             # Data collection utilities
├── models/
│   ├── attention.py          # Custom attention mechanisms
│   ├── proof_generator.py    # Main AVMCore model
│   └── strategies.py         # Proof strategy implementations
├── training/
│   ├── embeddings.py         # Embedding implementations
│   ├── optimizer.py          # Custom optimizers
│   ├── scheduler.py          # Learning rate schedulers
│   ├── train.py              # Training script
│   └── trainer.py            # Memory-efficient trainer
└── utils/
    ├── collate.py            # Batch collation utilities
    ├── evaluation.py         # Evaluation metrics
    ├── metrics.py            # Performance metrics
    ├── tokenizer.py          # Mathematical tokenizer
    └── visualization.py      # Visualization utilities
```

## Conclusion

The AVM-CORE architecture combines transformer-based sequence modeling with graph neural networks to understand and generate mathematical proofs. The system uses a multi-strategy approach to proof generation and includes components for validating the logical consistency of the generated proofs. The training pipeline is designed to be memory-efficient, allowing for training on consumer-grade hardware while maintaining model performance.