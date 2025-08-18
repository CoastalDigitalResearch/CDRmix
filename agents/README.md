# CDRmix Architecture Validation Agents

This directory contains specialized Claude Code agents for validating the CDRmix MoE-RWKV architecture.

## Available Agents

### 1. Neural-Math-Checker (`neural_math_checker.py`) 🚀 **PORTABLE**
Generic neural network mathematical validation agent that works across different architectures:
- **Multi-Architecture Support**: Transformer, RWKV, Mamba, MoE, and generic networks
- **Mathematical Validation**: Gradient flow, numerical stability, mathematical correctness
- **Auto-Detection**: Automatically detects architecture from codebase patterns
- **Cross-Project**: Designed to be copied to other ML projects (Mamba, etc.)

**Architecture-Specific Validation:**
- **Transformer**: Attention scaling, positional encoding, layer normalization
- **RWKV**: Linear complexity, recurrent formulation, streaming capabilities
- **Mamba**: State space model formulation, selective mechanisms, scan operations
- **MoE**: Expert routing, load balancing, capacity constraints
- **Generic**: Basic mathematical properties for any neural network

**Usage:**
```bash
# Auto-detect architecture and validate
python3 agents/neural_math_checker.py --project-root . --report

# Force specific architecture
python3 agents/neural_math_checker.py --architecture mamba --report

# Generate comprehensive test suite
python3 agents/neural_math_checker.py --generate-tests

# Quick validation summary
python3 agents/neural_math_checker.py
```

### 2. Reasoning-Validator (`reasoning_validator.py`) 🚀 **PORTABLE**
Generic reasoning system validation agent that works across different reasoning approaches:
- **Multi-Reasoning Support**: Chain-of-Thought, Tree-of-Thoughts, Self-Consistency, Hybrid Hebbian-RL, Memory-Augmented
- **Custom System Validation**: Specialized support for CDRmix Hybrid Hebbian-RL architecture
- **Memory System Analysis**: Working memory, episodic memory, retrieval systems, FAISS indexing
- **Controller Validation**: Branch management, proposal systems, evaluation criteria, rollback mechanisms
- **Plasticity Analysis**: Ephemeral adapters, Hebbian learning, correlation objectives, commit/rollback isolation
- **RL Policy Validation**: GRPO optimization, memory actions, advantage estimation, reward systems

**Reasoning-Specific Validation:**
- **Chain-of-Thought**: Step structure, reasoning validation, intermediate step verification
- **Tree-of-Thoughts**: Branch structure, evaluation mechanisms, pruning strategies
- **Hybrid Hebbian-RL**: Complete validation of CDRmix's custom reasoning architecture
- **Memory-Augmented**: Storage backends, retrieval mechanisms, context management
- **Generic**: Basic reasoning patterns for any system

**Usage:**
```bash
# Auto-detect reasoning systems and validate
python3 agents/reasoning_validator.py --project-root . --report

# Force specific reasoning types
python3 agents/reasoning_validator.py --reasoning-types hybrid_hebbian_rl chain_of_thought --report

# Generate comprehensive test suite
python3 agents/reasoning_validator.py --generate-tests

# Quick validation summary
python3 agents/reasoning_validator.py
```

### 3. MoE-Architecture-Validator (`moe_architecture_validator.py`)
Validates Mixture-of-Experts (MoE) architecture implementations with focus on:
- Expert routing algorithms and top-k constraints
- Load balancing and capacity constraints  
- Sparsity patterns and gradient scaling
- Compliance with formal Lean specifications
- Runtime validation of NoOverflow property

**Usage:**
```bash
# Generate validation report
python3 agents/moe_architecture_validator.py --project-root . --report

# Generate comprehensive test suite
python3 agents/moe_architecture_validator.py --project-root . --generate-tests

# Both report and tests
python3 agents/moe_architecture_validator.py --project-root . --report --generate-tests
```

### 2. RWKV-Block-Validator (`rwkv_block_validator.py`)
Validates RWKV (Receptance Weighted Key Value) block implementations with focus on:
- Mathematical property validation (R, W, K, V formulation)
- Linear-time complexity O(n) verification
- Recurrent formulation and state-based computation
- Streaming/incremental processing capabilities
- Attention-free operation validation
- Gradient flow and Lipschitz properties

**Usage:**
```bash
# Generate validation report  
python3 agents/rwkv_block_validator.py --project-root . --report

# Generate comprehensive test suite
python3 agents/rwkv_block_validator.py --project-root . --generate-tests

# Both report and tests
python3 agents/rwkv_block_validator.py --project-root . --report --generate-tests
```

## Generated Artifacts

### Validation Reports
- `moe_validation_report.md` - Detailed MoE architecture analysis
- `rwkv_validation_report.md` - Detailed RWKV block analysis

### Test Suites
- `test_moe_architecture.py` - Comprehensive MoE testing suite
- `test_rwkv_block.py` - Comprehensive RWKV testing suite

## Integration with Development Workflow

### 1. Pre-Implementation Validation
Run validators before implementing core components to understand requirements:
```bash
python3 agents/moe_architecture_validator.py --project-root . --report
python3 agents/rwkv_block_validator.py --project-root . --report
```

### 2. Post-Implementation Testing
After implementing components, run generated test suites:
```bash
pytest test_moe_architecture.py -v
pytest test_rwkv_block.py -v
```

### 3. CI/CD Integration
Add validation to your CI pipeline:
```yaml
- name: Validate MoE Architecture
  run: python3 agents/moe_architecture_validator.py --project-root . --report
  
- name: Validate RWKV Blocks  
  run: python3 agents/rwkv_block_validator.py --project-root . --report

- name: Run Architecture Tests
  run: |
    pytest test_moe_architecture.py -v
    pytest test_rwkv_block.py -v
```

## Current Validation Results

Based on the current CDRmix codebase (as of validation):

### Neural-Math-Checker Status: ❌ INVALID (MoE detected)
- **Issues:** 44 total (23 critical, 19 warnings, 2 info)
- **Architecture:** Auto-detected as MoE due to 'expert' keywords in codebase
- **Key Issues:** Missing top-k routing, load balancing, numerical stability mechanisms
- **Portability:** ✅ Successfully tested with different architecture flags

### Reasoning-Validator Status: ❌ INVALID (Multiple systems detected)
- **Issues:** 14 total (5 critical, 9 warnings)
- **Detected Systems:** Hybrid Hebbian-RL, Tree-of-Thoughts, Chain-of-Thought, Memory-Augmented, Generic
- **Key Issues:** Missing memory system implementation, plasticity system, RL policy components
- **Capabilities:** Excellent auto-detection of reasoning patterns from specifications
- **Portability:** ✅ Generic framework supports multiple reasoning approaches

### MoE Architecture Status: ❌ INVALID
- **Violations:** 3 (routing constraints, capacity limits, load balancing)
- **Warnings:** 10 (missing methods, normalization, expert dispatch)
- **Key Issues:** Placeholder implementations need proper MoE routing logic

### RWKV Block Status: ❌ INVALID  
- **Violations:** 9 (missing RWKV components, mathematical formulation)
- **Warnings:** 10 (linear complexity, streaming, gradient flow)
- **Key Issues:** Placeholder implementations need proper RWKV mathematical formulation

## Next Steps

1. **Implement Core Components:** Replace placeholder classes with proper MoE routing and RWKV mathematical formulations

2. **Address Violations:** Fix all reported violations before production deployment

3. **Run Test Suites:** Use generated test suites to validate implementations

4. **Monitor Runtime Properties:** Add runtime validation for capacity constraints and complexity properties

5. **Lean Specification Compliance:** Ensure implementations match formal mathematical specifications

## Agent Architecture

Both validators follow a common pattern:

1. **Static Code Analysis:** Parse implementation files and extract architectural patterns
2. **Mathematical Validation:** Check compliance with theoretical properties  
3. **Configuration Analysis:** Validate against YAML configs and Lean specifications
4. **Test Generation:** Create comprehensive test suites for runtime validation
5. **Report Generation:** Provide actionable validation reports

This provides both development-time validation and runtime testing capabilities for ensuring architectural correctness of the CDRmix MoE-RWKV model.

## Cross-Project Portability Guide

### Using Neural-Math-Checker in Other Projects

Both the `neural_math_checker.py` and `reasoning_validator.py` agents are designed to be portable across different projects. Here's how to use them in your Mamba or other language model projects:

#### 1. Copy the Agents
```bash
# Copy to your new project
cp agents/neural_math_checker.py /path/to/your/project/
cp agents/reasoning_validator.py /path/to/your/project/

# Make them executable
chmod +x neural_math_checker.py reasoning_validator.py
```

#### 2. Basic Usage in New Project
```bash
# Mathematical validation (auto-detect architecture)
python3 neural_math_checker.py --project-root . --report
python3 neural_math_checker.py --architecture mamba --report  # Force specific architecture

# Reasoning validation (auto-detect reasoning systems)
python3 reasoning_validator.py --project-root . --report
python3 reasoning_validator.py --reasoning-types chain_of_thought --report  # Force specific reasoning

# Generate test suites
python3 neural_math_checker.py --generate-tests
python3 reasoning_validator.py --generate-tests
```

#### 3. Supported Architectures & Reasoning Systems

**Mathematical Validation (neural_math_checker.py):**
- **Transformer**: Attention scaling, positional encoding validation
- **Mamba**: State space model, selective mechanism validation  
- **RWKV**: Linear complexity, recurrent formulation validation
- **MoE**: Expert routing, sparsity validation
- **Generic**: Basic mathematical properties for any neural network

**Reasoning Validation (reasoning_validator.py):**
- **Chain-of-Thought**: Step structure, reasoning validation
- **Tree-of-Thoughts**: Branch structure, evaluation mechanisms
- **Hybrid Hebbian-RL**: CDRmix custom reasoning architecture
- **Memory-Augmented**: Storage backends, retrieval systems
- **Self-Consistency**: Multiple reasoning path validation
- **Generic**: Basic reasoning patterns for any system

#### 4. Customization for New Systems
Extend validators for new architectures or reasoning systems:

**Mathematical Architecture Validator:**
```python
class YourArchitectureValidator(ArchitectureValidator):
    def get_architecture_type(self) -> ArchitectureType:
        return ArchitectureType.YOUR_ARCH  # Add to enum first
    
    def validate_architecture_specific_math(self, content: str, file_path: Path) -> List[MathValidationIssue]:
        # Your architecture-specific validation logic
        return issues
```

**Reasoning System Validator:**
```python
class YourReasoningValidator(ReasoningSystemValidator):
    def get_reasoning_type(self) -> ReasoningType:
        return ReasoningType.YOUR_REASONING  # Add to enum first
    
    def validate_reasoning_system(self, content: str, file_path: Path) -> List[ReasoningValidationIssue]:
        # Your reasoning-specific validation logic
        return issues
```

#### 5. Integration with Different Project Structures
The agent automatically searches common directories:
- `src/`, `models/`, `model/`, `lib/` for source files
- Auto-detects architecture from code patterns
- Generates reports and tests adapted to your project structure

#### 6. Continuous Integration
Add to your CI pipeline:
```yaml
- name: Mathematical Validation
  run: python3 neural_math_checker.py --project-root . --report
  
- name: Reasoning System Validation
  run: python3 reasoning_validator.py --project-root . --report
  
- name: Run Validation Tests  
  run: |
    pytest test_neural_math_validation.py -v
    pytest test_reasoning_validation.py -v
```

This makes both mathematical and reasoning validation standard parts of your development workflow across all neural network projects.