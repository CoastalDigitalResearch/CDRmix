# PRD: Reasoning Stack

## Product Overview

The CDRmix Reasoning Stack is a hybrid Hebbian-RL system that extends the core RWKX-V backbone with advanced reasoning capabilities. It enables models to perform multi-step reasoning through ephemeral plasticity adapters, memory management, and reinforcement learning-guided decision making. The system operates over four model scales (1B, 4B, 40B, 200B parameters) with automatically scaled reasoning parameters.

## Architecture Summary

The reasoning stack consists of three primary subsystems layered on top of the RWKX-V backbone:

### 1. Hybrid Hebbian-RL Reasoning Controller
- **Hebbian Proposer**: Generates reasoning proposals through ephemeral plasticity adapters (plastic-LoRA)
- **RL Evaluator (RL-MemAgent)**: Long-context policy over memory operations with GRPO-style learning
- **Decision System**: Branch scoring, acceptance/rejection, and rollback mechanisms

### 2. Memory Management System
- **Working Memory**: Streaming token-level state with KV/hidden summaries and configurable stride
- **Episodic Memory**: External FAISS-indexed storage with retrieval keys and experience traces
- **Memory Operations**: Summarize, retrieve, splice_window, cache_write actions

### 3. Plasticity & Adaptation Layer
- **Plastic-LoRA Adapters**: Low-rank adaptation matrices applied to attention and FFN layers
- **Local Correlation Objectives**: Hebbian learning with correlation-based loss functions
- **Ephemeral State Management**: Commit-on-accept policy with per-step decay

## Functional Requirements

### FR-1: Reasoning Controller Core
- **Requirement**: Hybrid system combining Hebbian plasticity with RL-based evaluation
- **Components**:
  - Hebbian proposer generating candidate reasoning paths through adapter modifications
  - RL-MemAgent policy network evaluating proposals based on task metrics, latency, and stability
  - Branch management system supporting concurrent proposals (max_branches: 3-4)
  - Accept/reject decision mechanism with configurable thresholds
- **Success Criteria**:
  - Support 3-4 concurrent reasoning branches per model scale
  - Accept/reject decisions based on advantage estimation ≥ threshold
  - Rollback capability for rejected proposals without state corruption

### FR-2: Memory Management Architecture  
- **Requirement**: Dual-memory system supporting both working and episodic memory
- **Working Memory Specifications**:
  - Streaming KV cache summarization with configurable stride (8 tokens)
  - Rolling summary dimensions scaling by model size (512/640/768/1024)
  - Token-level state preservation with compression
- **Episodic Memory Specifications**:
  - FAISS-indexed external storage for long-term experience
  - Configurable retrieval (top_k: 4-8 based on scale)
  - Maximum token windows scaling by size (16K/32K/131K/262K)
  - Write-on-accept policy preventing corruption from rejected branches

### FR-3: Plasticity Adaptation System
- **Requirement**: Ephemeral low-rank adapters enabling rapid model modification
- **Adapter Specifications**:
  - Plastic-LoRA application to attention projections (Q,K,V,O)
  - FFN/ChannelMix adaptation (when not using MoE)
  - Rank scaling by model size: 8/12/16/24
  - Alpha scaling: 0.2/0.18/0.16/0.15 (plasticity strength)
- **Learning Dynamics**:
  - Local correlation objectives (corr_loss_v1)
  - Per-step decay within episodes: 0.98/0.985/0.99/0.992
  - Commit-on-accept policy with gradient clipping (2.0)

### FR-4: RL-MemAgent Policy System
- **Requirement**: Reinforcement learning over memory operations with GRPO algorithm
- **Action Space**: [summarize, retrieve, splice_window, cache_write]
- **Reward Components**:
  - Task metric weight: 1.0 (primary objective)
  - Latency penalty: 0.01/0.01/0.008/0.006 (efficiency incentive)
  - Stability bonus: 0.1/0.1/0.12/0.12 (consistency reward)
- **Policy Optimization**:
  - PPO-style optimization with horizon scaling (256/384/512/768)
  - Discount factors: 0.995/0.996/0.997/0.998
  - KL divergence targets: 0.02/0.02/0.02/0.015
  - Clip ratios: 0.2 (consistent across scales)

### FR-5: Scale-Adaptive Parameter Management
Reasoning parameters automatically scale with model size:

| Parameter | 1B | 4B | 40B | 200B |
|-----------|----|----|-----|------|
| **Controller** |
| max_branches | 3 | 3 | 4 | 4 |
| step_budget | 64 | 96 | 128 | 160 |
| accept_threshold | 0.0 | 0.0 | 0.0 | 0.0 |
| **Hebbian Adapters** |
| rank | 8 | 12 | 16 | 24 |
| alpha | 0.2 | 0.18 | 0.16 | 0.15 |
| decay | 0.98 | 0.985 | 0.99 | 0.992 |
| **RL Policy** |
| horizon | 256 | 384 | 512 | 768 |
| gamma | 0.995 | 0.996 | 0.997 | 0.998 |
| kl_target | 0.02 | 0.02 | 0.02 | 0.015 |
| **Memory** |
| summary_dim | 512 | 640 | 768 | 1024 |
| episodic_top_k | 4 | 6 | 8 | 8 |
| max_tokens | 16384 | 32768 | 131072 | 262144 |

## Technical Requirements

### TR-1: Memory Efficiency
- **Working Memory**: Configurable summarization stride to balance context retention vs memory usage
- **Episodic Storage**: External FAISS indexing to prevent GPU memory overflow
- **Adapter Storage**: Ephemeral LoRA parameters with automatic cleanup on branch rejection
- **Large Scale Support**: Sequence checkpointing for 40B/200B models with long reasoning horizons

### TR-2: Training Stability  
- **Loss Composition**: Balanced multi-objective training [ce, rl, hebbian, moe_aux]
- **Gradient Management**: Model-scale-specific clipping (1.0/1.0/0.9/0.8) and initialization scaling
- **Policy Stability**: KL divergence monitoring with automatic learning rate adjustment
- **Branch Management**: Proper state isolation preventing cross-contamination

### TR-3: Reasoning Performance
- **Latency Targets**: 
  - Memory operations: <10ms per action
  - Branch evaluation: <50ms per proposal  
  - Adapter updates: <5ms per modification
- **Throughput Scaling**: Concurrent branch processing without blocking
- **Memory Retrieval**: Sub-linear scaling with episodic memory size through FAISS optimization

### TR-4: Integration Architecture
- **Backbone Compatibility**: Clean integration with RWKX-V layers without architectural conflicts
- **MoE Interaction**: Proper routing with reasoning-aware expert selection
- **Hardware Support**: Memory management compatible with distributed training across multiple nodes
- **State Serialization**: Checkpoint compatibility including adapter states and memory indices

## Implementation Architecture

### Component Structure

**Reasoning Controller** (`reasoning/`)
- `adapters/`: Plastic-LoRA implementation and adapter management
- `memagent/`: RL policy networks, reward calculation, and GRPO optimization
- `memory/`: Working memory summarization and episodic memory indexing

**Memory Operations** (`reasoning/memory/`)
```python
class MemoryOperations:
    def summarize(self, context_window: Tensor) -> Tensor
    def retrieve(self, query: Tensor, top_k: int) -> List[Tensor]  
    def splice_window(self, position: int, content: Tensor) -> None
    def cache_write(self, key: Tensor, value: Tensor) -> None
```

**Hebbian Plasticity** (`reasoning/adapters/`)
```python
class PlasticLoRA:
    def __init__(self, rank: int, alpha: float, target_modules: List[str])
    def forward(self, x: Tensor, adapt: bool = True) -> Tensor
    def compute_correlation_loss(self, activations: Tensor) -> Tensor
    def commit_adapters(self) -> None
    def rollback_adapters(self) -> None
```

### Reasoning Flow

```mermaid
graph TD
    A[Input Token Sequence] --> B[RWKX-V Backbone Processing]
    B --> C[Reasoning Controller]
    C --> D[Hebbian Proposer]
    D --> E[Generate Branch Candidates]
    E --> F[RL-MemAgent Evaluation]
    F --> G{Accept Branch?}
    G -->|Yes| H[Commit Adapters]
    G -->|No| I[Rollback State]
    H --> J[Update Memory]
    I --> K[Try Next Branch]
    J --> L[Continue Generation]
    K --> L
    
    subgraph Memory System
        M1[Working Memory]
        M2[Episodic Memory]
        M3[Memory Operations]
    end
    
    C <--> Memory System
```

### Training Integration

**Stage B Training Loop** (`training/train_stageB.py`)
1. Load pretrained backbone from Stage A
2. Initialize reasoning components with scale-appropriate parameters
3. Multi-objective loss optimization:
   - Cross-entropy loss (language modeling)
   - RL rewards (memory operation efficiency)
   - Hebbian correlation loss (plasticity objectives)
   - MoE auxiliary losses (routing balance)
4. Policy gradient updates with GRPO algorithm
5. Adapter commitment/rollback based on advantage estimation

## Success Metrics

### Reasoning Capability Metrics
- **Multi-Step Reasoning**: >90% accuracy on chain-of-thought benchmarks
- **Memory Utilization**: Efficient retrieval with <5% memory overhead
- **Branch Efficiency**: >70% branch acceptance rate (not too conservative/aggressive)
- **Adaptation Speed**: Convergence within 10-20 reasoning steps per problem

### Performance Metrics
- **Latency Overhead**: <30% increase vs backbone-only inference
- **Memory Scaling**: Linear growth with episodic memory size (not quadratic)
- **Training Stability**: Converge within 2x backbone training steps
- **Policy Learning**: RL rewards show consistent improvement over training

### System Metrics
- **Memory Efficiency**: <20% additional GPU memory usage for reasoning components
- **Fault Tolerance**: Successful recovery from branch evaluation failures
- **Scalability**: Consistent reasoning quality across all four model scales
- **Integration**: No degradation of core language modeling performance

## Dependencies

### Core Dependencies
- **Backbone**: Pretrained RWKX-V models from Stage A training
- **Memory Backend**: FAISS library for episodic memory indexing
- **RL Framework**: Integration with PPO/GRPO implementations
- **Adapter Framework**: LoRA/plastic adaptation libraries

### Data Requirements
- **Reasoning Datasets**: Chain-of-thought, mathematical reasoning, logical inference tasks
- **Memory Training Data**: Long-context documents for episodic memory population
- **Policy Training**: Reward signal datasets for memory operation optimization

### Hardware Requirements
- **Memory Scaling**: Additional 20-30% GPU memory for reasoning components
- **Compute Overhead**: 2x training time vs backbone-only for Stage B
- **Storage**: External episodic memory requires high-speed storage access
- **Network**: Distributed reasoning requires low-latency inter-node communication

## Validation Plan

### Component Testing
1. **Hebbian Adapters**: Validate correlation objectives and adapter commitment/rollback
2. **Memory Operations**: Test working memory summarization and episodic retrieval accuracy  
3. **RL Policy**: Verify GRPO convergence and reward optimization
4. **Integration**: End-to-end reasoning pipeline with backbone compatibility

### Reasoning Benchmarks
- **Mathematical Reasoning**: GSM8K, MATH benchmark performance
- **Logical Reasoning**: LogiQA, HellaSwag with multi-step inference
- **Long-Context**: Reasoning over 10K+ token contexts with memory utilization
- **Tool Usage**: API integration for external memory and computation tools

### Scale Validation
- [ ] 1B model: Baseline reasoning capability with minimal overhead
- [ ] 4B model: Enhanced reasoning with improved memory utilization
- [ ] 40B model: Complex multi-step reasoning with distributed memory
- [ ] 200B model: Advanced reasoning matching human-level performance on benchmarks

## Risk Mitigation

### Technical Risks
- **Memory Overflow**: Implement adaptive memory compression and cleanup strategies
- **Training Instability**: Careful loss weighting and gradient clipping across objectives
- **Policy Collapse**: KL divergence monitoring with automatic rollback mechanisms
- **Adapter Interference**: Proper isolation between concurrent reasoning branches

### Performance Risks  
- **Latency Degradation**: Asynchronous memory operations and speculative branch evaluation
- **Memory Scaling**: Hierarchical memory with automatic archiving of old episodes
- **Integration Complexity**: Modular architecture enabling independent component development

### Resource Risks
- **Compute Requirements**: Progressive scaling validation to identify resource bottlenecks early
- **Storage Needs**: Distributed episodic memory with automatic garbage collection
- **Development Complexity**: Clear API boundaries and comprehensive testing frameworks