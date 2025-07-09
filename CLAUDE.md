# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CDRmix is an open-source Mixture-of-Experts (MoE) large language model architecture built on RWKV-style blocks. It's designed for streaming-capable, long-context reasoning with efficient deployment across heterogeneous hardware platforms (NVIDIA, AMD, Tenstorrent).

## Core Architecture

### Model Structure
- **RWKV-based experts** for linear-time, streaming compatibility
- **Sparse MoE routing** with top-k expert selection
- **Hardware abstraction layer** supporting CUDA, ROCm, and Tenstorrent
- **Token state-based memory** for low-VRAM edge devices

### Key Components
- `src/cdrmix/` - Core model architecture (RWKV blocks, MoE logic)
- `src/backends/` - Hardware backend abstractions (torch_backend.py, rocm_backend.py, tenstorrent_backend.py)
- `config/` - YAML-based configuration system for models, training, and evaluation
- `tools/` - Data processing utilities for dataset preparation

## Development Commands

### Training
```bash
# Train 1B model with default config
python3 src/train.py --config config/train-1b.yaml

# Train via shell script
./scripts/train.sh
```

### Evaluation
```bash
# Evaluate model checkpoint
python3 src/eval.py --config config/eval-1b.yaml --checkpoint checkpoints/model.pt

# Evaluate via shell script
./scripts/eval.sh
```

### Data Processing
```bash
# Convert directory structure to JSONL format
python3 tools/directory_to_jsonl.py <input_dir> <output_file>

# Create evaluation dataset from training data
python3 tools/make_eval_dataset.py <input_jsonl> <output_jsonl> --split_ratio 0.1
```

## Configuration System

### Model Configurations
- `config/train-1b.yaml` - 1B parameter model training config
- `config/eval-1b.yaml` - 1B parameter model evaluation config
- `config/config-1.5b.yaml`, `config/config-4b.yaml`, `config/config-8b.yaml` - Model size variants

### Key Configuration Parameters
- `num_layers`: Number of transformer layers
- `hidden_size`: Hidden dimension size
- `moe_every`: Insert MoE block every N layers
- `num_experts`: Total number of experts in MoE blocks
- `top_k`: Number of experts to activate per token
- `seq_length`: Maximum sequence length (default: 2048)

## Dataset Requirements

### Training Data
- Primary dataset: Common Pile v0.1 (Filtered)
- Format: Tokenized JSONL files in sharded structure
- Path structure: `/mnt/data/tokenized_shards/**/*.jsonl`
- Tokenizer: BPE-32k located at `/mnt/tokenizer/cdrmix-bpe-32k`

### Data Pipeline
- Streaming JSONL reader for memory efficiency
- Configurable sequence length and batching
- Support for multiple dataset sources in configuration

## Hardware Support

### Backend Abstraction
- **PyTorch/CUDA**: Primary development and training backend
- **ROCm**: AMD GPU support (planned)
- **Tenstorrent**: Specialized AI hardware support (planned)

### Device Selection
- Automatic CUDA detection with CPU fallback
- Hardware-specific optimizations through backend system

## Development Status

### Current State
- **1B Model**: In active development/training
- **Core Architecture**: Implemented but some components are placeholders
- **Data Pipeline**: Functional with robust tooling
- **Testing**: No test infrastructure currently exists

### Known Limitations
- Many core model components are placeholder implementations
- No formal dependency management (requirements.txt/pyproject.toml)
- Missing comprehensive testing suite
- No CI/CD pipeline established

## Key File Locations

### Core Implementation
- `src/cdrmix/cdrmix_model.py` - Main model class
- `src/cdrmix/rwkv_block.py` - RWKV block implementation
- `src/cdrmix/moe_block.py` - MoE block implementation
- `src/cdrmix/router.py` - Expert routing logic

### Training Pipeline
- `src/train.py` - Main training loop
- `src/eval.py` - Model evaluation
- `src/data_loader.py` - Dataset loading pipeline

### Utilities
- `src/utils/config.py` - YAML configuration loader
- `src/utils/tokenizer_utils.py` - Tokenizer utilities
- `src/export.py` - Model export utilities (stub)

## Development Priorities

When working on this codebase:

1. **Fill placeholder implementations** in core model components
2. **Add comprehensive testing** following TDD principles
3. **Implement proper dependency management** with requirements.txt or pyproject.toml
4. **Add type hints** throughout the codebase
5. **Set up CI/CD pipeline** for automated testing and validation

## Important Notes

- Project uses MIT license - ensure all contributions are compatible
- Focus on streaming capabilities and memory efficiency
- Maintain hardware abstraction for cross-platform deployment
- Follow YAML-based configuration patterns for consistency
- All training data should be permissively licensed