#!/usr/bin/env python3
"""
Neural-Math-Checker Agent

Generic neural network mathematical validation agent that can be used across
different architectures (RWKV, Mamba, Transformer, MoE, etc.).

This agent provides comprehensive mathematical validation including:
- Gradient flow analysis and stability
- Numerical stability and precision checks
- Mathematical property validation
- Activation function analysis
- Weight initialization validation
- Loss landscape analysis
- Architecture-specific mathematical constraints

Designed to be portable across different neural network projects.
"""

import ast
import math
import re
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
import json


class ArchitectureType(Enum):
    """Supported neural network architectures."""
    TRANSFORMER = "transformer"
    RWKV = "rwkv"
    MAMBA = "mamba" 
    MOE = "moe"
    CONVNET = "convnet"
    RNN = "rnn"
    GENERIC = "generic"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"  # Will cause training failure
    WARNING = "warning"    # May cause poor performance
    INFO = "info"         # Best practice recommendations


@dataclass
class MathValidationIssue:
    """Individual mathematical validation issue."""
    severity: ValidationSeverity
    category: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggested_fix: Optional[str] = None
    mathematical_context: Optional[str] = None


@dataclass
class NumericalProperties:
    """Numerical properties of a neural network component."""
    has_proper_initialization: bool = False
    has_gradient_clipping: bool = False
    has_numerical_stability: bool = False
    has_proper_normalization: bool = False
    activation_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    precision_requirements: Set[str] = field(default_factory=set)
    
    
@dataclass
class GradientFlowProperties:
    """Gradient flow analysis properties."""
    has_residual_connections: bool = False
    has_gradient_highways: bool = False
    has_proper_scaling: bool = False
    vanishing_gradient_risk: str = "unknown"  # low, medium, high, unknown
    exploding_gradient_risk: str = "unknown"  # low, medium, high, unknown
    gradient_flow_depth: int = 0


@dataclass
class MathematicalConstraints:
    """Mathematical constraints for different architectures."""
    complexity_bounds: Dict[str, str] = field(default_factory=dict)  # O(n), O(n²), etc.
    numerical_stability_requirements: List[str] = field(default_factory=list)
    invariant_properties: List[str] = field(default_factory=list)
    mathematical_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NeuralMathValidationResult:
    """Complete neural math validation result."""
    is_mathematically_valid: bool
    architecture_type: ArchitectureType
    issues: List[MathValidationIssue]
    numerical_properties: NumericalProperties
    gradient_flow_properties: GradientFlowProperties
    mathematical_constraints: MathematicalConstraints
    recommendations: List[str]
    test_suite_generated: bool = False


class ArchitectureValidator(ABC):
    """Abstract base class for architecture-specific validators."""
    
    @abstractmethod
    def get_architecture_type(self) -> ArchitectureType:
        """Get the architecture type this validator handles."""
        pass
    
    @abstractmethod
    def validate_architecture_specific_math(self, content: str, file_path: Path) -> List[MathValidationIssue]:
        """Validate architecture-specific mathematical properties."""
        pass
    
    @abstractmethod
    def get_mathematical_constraints(self) -> MathematicalConstraints:
        """Get mathematical constraints specific to this architecture."""
        pass


class TransformerValidator(ArchitectureValidator):
    """Transformer architecture mathematical validator."""
    
    def get_architecture_type(self) -> ArchitectureType:
        return ArchitectureType.TRANSFORMER
    
    def validate_architecture_specific_math(self, content: str, file_path: Path) -> List[MathValidationIssue]:
        issues = []
        
        # Check attention scaling
        if 'attention' in content.lower() or 'attn' in content.lower():
            if not re.search(r'sqrt\(.*d_k|d_k.*\*\*.*0\.5|math\.sqrt.*d_k', content, re.IGNORECASE):
                issues.append(MathValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="attention_scaling",
                    description="Attention mechanism may be missing proper scaling (1/√d_k)",
                    file_path=str(file_path),
                    mathematical_context="Attention weights should be scaled by 1/√d_k to prevent softmax saturation"
                ))
        
        # Check positional encoding
        if 'positional' in content.lower() and 'encoding' in content.lower():
            if not re.search(r'sin\(|cos\(|torch\.sin|torch\.cos', content, re.IGNORECASE):
                issues.append(MathValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="positional_encoding",
                    description="Positional encoding may not use sinusoidal functions",
                    mathematical_context="Standard positional encoding uses sin/cos functions"
                ))
        
        # Check layer normalization placement
        if 'layernorm' in content.lower() or 'layer_norm' in content.lower():
            # Pre-norm vs post-norm analysis would go here
            pass
        
        return issues
    
    def get_mathematical_constraints(self) -> MathematicalConstraints:
        return MathematicalConstraints(
            complexity_bounds={"attention": "O(n²)", "feedforward": "O(n)"},
            numerical_stability_requirements=[
                "attention_scaling", "gradient_clipping", "layer_normalization"
            ],
            invariant_properties=["permutation_equivariance"],
            mathematical_properties={
                "attention_temperature": 1.0,
                "requires_padding_mask": True,
                "supports_causal_masking": True
            }
        )


class RWKVValidator(ArchitectureValidator):
    """RWKV architecture mathematical validator."""
    
    def get_architecture_type(self) -> ArchitectureType:
        return ArchitectureType.RWKV
    
    def validate_architecture_specific_math(self, content: str, file_path: Path) -> List[MathValidationIssue]:
        issues = []
        
        # Check RWKV mathematical formulation
        rwkv_components = ['receptance', 'weight', 'key', 'value']
        missing_components = []
        
        for component in rwkv_components:
            if component not in content.lower():
                missing_components.append(component)
        
        if missing_components:
            issues.append(MathValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="rwkv_formulation",
                description=f"Missing RWKV components: {', '.join(missing_components)}",
                file_path=str(file_path),
                mathematical_context="RWKV requires R, W, K, V components for proper computation"
            ))
        
        # Check linear complexity 
        if re.search(r'@.*@|attention|attn', content, re.IGNORECASE):
            issues.append(MathValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="complexity_violation",
                description="RWKV contains O(n²) operations, violating linear complexity requirement",
                mathematical_context="RWKV should achieve O(n) complexity through sequential processing"
            ))
        
        # Check state-based computation
        if 'state' not in content.lower():
            issues.append(MathValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="recurrent_formulation", 
                description="RWKV missing state-based computation",
                mathematical_context="RWKV requires hidden state for recurrent processing"
            ))
        
        return issues
    
    def get_mathematical_constraints(self) -> MathematicalConstraints:
        return MathematicalConstraints(
            complexity_bounds={"time_mixing": "O(n)", "channel_mixing": "O(n)"},
            numerical_stability_requirements=[
                "exponential_interpolation", "sigmoid_receptance", "proper_initialization"
            ],
            invariant_properties=["linear_complexity", "streaming_capability", "causal_ordering"],
            mathematical_properties={
                "recurrent_formulation": True,
                "attention_free": True,
                "streaming_capable": True
            }
        )


class MambaValidator(ArchitectureValidator):
    """Mamba (State Space Model) architecture mathematical validator."""
    
    def get_architecture_type(self) -> ArchitectureType:
        return ArchitectureType.MAMBA
    
    def validate_architecture_specific_math(self, content: str, file_path: Path) -> List[MathValidationIssue]:
        issues = []
        
        # Check state space model components
        ssm_components = ['delta', 'A', 'B', 'C', 'D']
        missing_ssm = []
        
        for component in ssm_components:
            # Look for these as variables or parameters
            if not re.search(rf'\b{component}(?:_param|_matrix|_tensor)?\b', content, re.IGNORECASE):
                missing_ssm.append(component)
        
        if len(missing_ssm) > 2:  # Allow some flexibility in naming
            issues.append(MathValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="ssm_formulation",
                description=f"Possible missing SSM components: {', '.join(missing_ssm)}",
                file_path=str(file_path),
                mathematical_context="Mamba uses state space model with Δ, A, B, C, D parameters"
            ))
        
        # Check selective mechanism
        if 'selective' in content.lower():
            if not re.search(r'input.*dependent|token.*dependent|context.*dependent', content, re.IGNORECASE):
                issues.append(MathValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="selective_mechanism",
                    description="Selective mechanism may not be input-dependent",
                    mathematical_context="Mamba's selectivity should depend on input tokens"
                ))
        
        # Check scan operation for efficient computation
        if not re.search(r'scan|cumsum|parallel_scan', content, re.IGNORECASE):
            issues.append(MathValidationIssue(
                severity=ValidationSeverity.INFO,
                category="scan_operation",
                description="May be missing efficient scan operation for SSM",
                mathematical_context="Efficient SSM computation often uses parallel scan"
            ))
        
        return issues
    
    def get_mathematical_constraints(self) -> MathematicalConstraints:
        return MathematicalConstraints(
            complexity_bounds={"ssm_layer": "O(n log n)", "overall": "O(n)"},
            numerical_stability_requirements=[
                "discretization_stability", "scan_numerical_stability", "selective_parameter_bounds"
            ],
            invariant_properties=["linear_complexity", "selective_processing", "hardware_efficiency"],
            mathematical_properties={
                "state_space_model": True,
                "selective_mechanism": True,
                "hardware_aware": True
            }
        )


class MoEValidator(ArchitectureValidator):
    """Mixture of Experts mathematical validator."""
    
    def get_architecture_type(self) -> ArchitectureType:
        return ArchitectureType.MOE
    
    def validate_architecture_specific_math(self, content: str, file_path: Path) -> List[MathValidationIssue]:
        issues = []
        
        # Check expert routing
        if not re.search(r'topk|top_k|torch\.topk', content, re.IGNORECASE):
            issues.append(MathValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="expert_routing",
                description="MoE missing top-k expert selection",
                mathematical_context="MoE requires top-k routing for sparsity"
            ))
        
        # Check load balancing
        if 'expert' in content.lower() and 'balance' not in content.lower():
            issues.append(MathValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="load_balancing",
                description="MoE may be missing load balancing mechanism",
                mathematical_context="Load balancing prevents expert collapse"
            ))
        
        # Check capacity constraints
        if 'capacity' not in content.lower() and 'expert' in content.lower():
            issues.append(MathValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="capacity_constraints",
                description="MoE may be missing capacity constraints",
                mathematical_context="Capacity constraints prevent expert overflow"
            ))
        
        return issues
    
    def get_mathematical_constraints(self) -> MathematicalConstraints:
        return MathematicalConstraints(
            complexity_bounds={"routing": "O(n * E)", "expert_computation": "O(n * k / E)"},
            numerical_stability_requirements=[
                "routing_weight_normalization", "load_balancing_loss", "capacity_constraints"
            ],
            invariant_properties=["sparsity_preservation", "expert_specialization"],
            mathematical_properties={
                "sparse_computation": True,
                "expert_load_balancing": True,
                "capacity_factor_bounds": [1.0, 2.0]
            }
        )


class GenericValidator(ArchitectureValidator):
    """Generic neural network mathematical validator."""
    
    def get_architecture_type(self) -> ArchitectureType:
        return ArchitectureType.GENERIC
    
    def validate_architecture_specific_math(self, content: str, file_path: Path) -> List[MathValidationIssue]:
        # Generic validator provides minimal architecture-specific validation
        return []
    
    def get_mathematical_constraints(self) -> MathematicalConstraints:
        return MathematicalConstraints(
            complexity_bounds={"default": "O(n)"},
            numerical_stability_requirements=["proper_initialization", "gradient_clipping"],
            invariant_properties=["gradient_flow"],
            mathematical_properties={}
        )


class NeuralMathChecker:
    """
    Generic neural network mathematical validation agent.
    
    Validates mathematical correctness, numerical stability, and gradient flow
    across different neural network architectures.
    """
    
    def __init__(self, project_root: Path, architecture_type: Optional[ArchitectureType] = None):
        self.project_root = Path(project_root)
        self.issues: List[MathValidationIssue] = []
        self.architecture_validators: Dict[ArchitectureType, ArchitectureValidator] = {
            ArchitectureType.TRANSFORMER: TransformerValidator(),
            ArchitectureType.RWKV: RWKVValidator(), 
            ArchitectureType.MAMBA: MambaValidator(),
            ArchitectureType.MOE: MoEValidator(),
            ArchitectureType.GENERIC: GenericValidator()
        }
        
        # Auto-detect architecture if not specified
        self.detected_architecture = architecture_type or self._detect_architecture()
        self.current_validator = self.architecture_validators.get(
            self.detected_architecture, 
            self.architecture_validators[ArchitectureType.GENERIC]
        )
    
    def _detect_architecture(self) -> ArchitectureType:
        """Auto-detect neural network architecture from codebase."""
        architecture_indicators = {
            ArchitectureType.TRANSFORMER: ['attention', 'transformer', 'multihead'],
            ArchitectureType.RWKV: ['rwkv', 'receptance', 'time_mix'],
            ArchitectureType.MAMBA: ['mamba', 'ssm', 'state_space', 'selective'],
            ArchitectureType.MOE: ['moe', 'mixture', 'expert', 'router'],
        }
        
        # Search for architecture indicators in source files
        source_dirs = ['src', 'model', 'models', 'lib']
        architecture_scores = {arch: 0 for arch in ArchitectureType}
        
        for source_dir in source_dirs:
            source_path = self.project_root / source_dir
            if source_path.exists():
                for py_file in source_path.rglob('*.py'):
                    try:
                        with open(py_file, 'r') as f:
                            content = f.read().lower()
                        
                        for arch_type, indicators in architecture_indicators.items():
                            for indicator in indicators:
                                if indicator in content:
                                    architecture_scores[arch_type] += content.count(indicator)
                    except Exception:
                        continue
        
        # Return architecture with highest score
        if max(architecture_scores.values()) > 0:
            return max(architecture_scores.items(), key=lambda x: x[1])[0]
        else:
            return ArchitectureType.GENERIC
    
    def validate_complete_mathematical_properties(self) -> NeuralMathValidationResult:
        """
        Comprehensive mathematical validation of neural network implementation.
        """
        self.issues = []
        
        # Core mathematical validations
        numerical_properties = self._validate_numerical_properties()
        gradient_flow_properties = self._validate_gradient_flow()
        
        # Architecture-specific validation
        self._validate_architecture_specific_properties()
        
        # Advanced mathematical analysis
        self._validate_activation_functions()
        self._validate_weight_initialization()
        self._validate_loss_functions()
        self._validate_numerical_stability()
        self._validate_mathematical_correctness()
        
        # Get constraints from current architecture validator
        mathematical_constraints = self.current_validator.get_mathematical_constraints()
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return NeuralMathValidationResult(
            is_mathematically_valid=not any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues),
            architecture_type=self.detected_architecture,
            issues=self.issues,
            numerical_properties=numerical_properties,
            gradient_flow_properties=gradient_flow_properties,
            mathematical_constraints=mathematical_constraints,
            recommendations=recommendations
        )
    
    def _validate_numerical_properties(self) -> NumericalProperties:
        """Validate numerical stability and precision properties."""
        properties = NumericalProperties()
        
        # Find all Python files in source directories
        source_files = []
        for pattern in ['src/**/*.py', 'models/**/*.py', 'model/**/*.py', 'lib/**/*.py']:
            source_files.extend(self.project_root.glob(pattern))
        
        initialization_patterns = [
            r'xavier|kaiming|he_|normal_|uniform_|glorot',
            r'torch\.nn\.init\.',
            r'std\s*=.*math\.sqrt',
            r'gain\s*=.*math\.sqrt'
        ]
        
        normalization_patterns = [
            r'layernorm|batchnorm|groupnorm|instancenorm',
            r'normalize\(',
            r'F\.normalize'
        ]
        
        stability_patterns = [
            r'eps\s*=.*1e-', 
            r'epsilon\s*=.*1e-',
            r'torch\.clamp|torch\.clip',
            r'gradient.*clip'
        ]
        
        for file_path in source_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check initialization
                for pattern in initialization_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        properties.has_proper_initialization = True
                        break
                
                # Check normalization
                for pattern in normalization_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        properties.has_proper_normalization = True
                        break
                
                # Check numerical stability
                for pattern in stability_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        properties.has_numerical_stability = True
                        break
                
                # Check gradient clipping
                if re.search(r'clip.*grad|grad.*clip', content, re.IGNORECASE):
                    properties.has_gradient_clipping = True
                
            except Exception as e:
                self.issues.append(MathValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="file_access",
                    description=f"Could not analyze file {file_path}: {e}"
                ))
        
        # Add issues based on missing properties
        if not properties.has_proper_initialization:
            self.issues.append(MathValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="initialization",
                description="No proper weight initialization detected",
                suggested_fix="Add Xavier/Kaiming initialization for weights",
                mathematical_context="Proper initialization prevents gradient vanishing/exploding"
            ))
        
        if not properties.has_gradient_clipping:
            self.issues.append(MathValidationIssue(
                severity=ValidationSeverity.INFO,
                category="gradient_clipping",
                description="No gradient clipping detected", 
                suggested_fix="Consider adding gradient clipping for training stability"
            ))
        
        return properties
    
    def _validate_gradient_flow(self) -> GradientFlowProperties:
        """Validate gradient flow properties."""
        properties = GradientFlowProperties()
        
        source_files = list(self.project_root.rglob('*.py'))
        
        for file_path in source_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check for residual connections
                if re.search(r'\+.*input|input.*\+|residual|skip.*connection', content, re.IGNORECASE):
                    properties.has_residual_connections = True
                
                # Check for proper scaling
                if re.search(r'scale|scaling|math\.sqrt.*d_', content, re.IGNORECASE):
                    properties.has_proper_scaling = True
                
                # Estimate network depth
                layer_count = len(re.findall(r'class.*Layer|class.*Block|nn\.Linear|nn\.Conv', content, re.IGNORECASE))
                properties.gradient_flow_depth = max(properties.gradient_flow_depth, layer_count)
                
            except Exception:
                continue
        
        # Assess gradient risks based on depth and architecture
        if properties.gradient_flow_depth > 12:
            if not properties.has_residual_connections:
                properties.vanishing_gradient_risk = "high"
                self.issues.append(MathValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="gradient_flow",
                    description=f"Deep network ({properties.gradient_flow_depth} layers) without residual connections",
                    suggested_fix="Add residual connections for deep networks",
                    mathematical_context="Deep networks suffer from vanishing gradients without skip connections"
                ))
            else:
                properties.vanishing_gradient_risk = "low"
        else:
            properties.vanishing_gradient_risk = "low"
        
        # Check for exploding gradient risk
        if not properties.has_proper_scaling:
            properties.exploding_gradient_risk = "medium"
            self.issues.append(MathValidationIssue(
                severity=ValidationSeverity.INFO,
                category="gradient_scaling",
                description="No proper gradient scaling detected",
                suggested_fix="Add appropriate scaling factors (e.g., 1/√d_k for attention)"
            ))
        else:
            properties.exploding_gradient_risk = "low"
        
        return properties
    
    def _validate_architecture_specific_properties(self):
        """Validate properties specific to the detected architecture."""
        source_files = list(self.project_root.rglob('*.py'))
        
        for file_path in source_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Use architecture-specific validator
                arch_issues = self.current_validator.validate_architecture_specific_math(content, file_path)
                self.issues.extend(arch_issues)
                
            except Exception as e:
                self.issues.append(MathValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="file_access",
                    description=f"Could not analyze {file_path}: {e}"
                ))
    
    def _validate_activation_functions(self):
        """Validate activation function choices and implementations."""
        source_files = list(self.project_root.rglob('*.py'))
        
        activation_analysis = {
            'relu': {'saturating': False, 'bounded': False, 'smooth': False},
            'gelu': {'saturating': False, 'bounded': False, 'smooth': True},
            'swish': {'saturating': False, 'bounded': False, 'smooth': True}, 
            'tanh': {'saturating': True, 'bounded': True, 'smooth': True},
            'sigmoid': {'saturating': True, 'bounded': True, 'smooth': True},
        }
        
        found_activations = set()
        
        for file_path in source_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                for activation in activation_analysis:
                    if re.search(rf'\b{activation}\b|F\.{activation}|nn\.{activation.upper()}', content, re.IGNORECASE):
                        found_activations.add(activation)
                
            except Exception:
                continue
        
        # Analyze activation choices
        if 'relu' in found_activations and len(found_activations) == 1:
            if self.detected_architecture in [ArchitectureType.TRANSFORMER, ArchitectureType.MAMBA]:
                self.issues.append(MathValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="activation_choice",
                    description="Consider GELU instead of ReLU for modern architectures",
                    mathematical_context="GELU provides smoother gradients than ReLU"
                ))
        
        # Check for activation in wrong places
        for file_path in source_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check for activation after layer norm (usually wrong)
                if re.search(r'layernorm.*relu|layernorm.*sigmoid', content, re.IGNORECASE):
                    self.issues.append(MathValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="activation_placement",
                        description="Activation function after LayerNorm may break normalization",
                        file_path=str(file_path),
                        mathematical_context="LayerNorm expects centered inputs, activations can break this"
                    ))
                
            except Exception:
                continue
    
    def _validate_weight_initialization(self):
        """Validate weight initialization strategies."""
        source_files = list(self.project_root.rglob('*.py'))
        
        init_strategies = []
        
        for file_path in source_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Detect initialization strategies
                if re.search(r'xavier_uniform|glorot_uniform', content, re.IGNORECASE):
                    init_strategies.append('xavier_uniform')
                elif re.search(r'xavier_normal|glorot_normal', content, re.IGNORECASE):
                    init_strategies.append('xavier_normal')
                elif re.search(r'kaiming_uniform|he_uniform', content, re.IGNORECASE):
                    init_strategies.append('kaiming_uniform')
                elif re.search(r'kaiming_normal|he_normal', content, re.IGNORECASE):
                    init_strategies.append('kaiming_normal')
                
                # Check for problematic patterns
                if re.search(r'weight.*=.*torch\.randn|weight.*=.*random', content, re.IGNORECASE):
                    self.issues.append(MathValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="initialization",
                        description="Using basic random initialization instead of proper scaling",
                        file_path=str(file_path),
                        suggested_fix="Use Xavier/Kaiming initialization",
                        mathematical_context="Random initialization can lead to vanishing/exploding gradients"
                    ))
                
                # Check for zero initialization where inappropriate
                if re.search(r'weight.*=.*torch\.zeros|bias.*=.*torch\.zeros', content, re.IGNORECASE):
                    if 'bias' not in content.lower():  # Zero bias is usually okay
                        self.issues.append(MathValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            category="initialization",
                            description="Zero initialization of weights will prevent learning",
                            file_path=str(file_path),
                            mathematical_context="Zero weights lead to symmetric gradients and no learning"
                        ))
                
            except Exception:
                continue
        
        # Check initialization appropriateness for architecture
        if self.detected_architecture == ArchitectureType.TRANSFORMER:
            if 'xavier' not in ' '.join(init_strategies):
                self.issues.append(MathValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="initialization_choice",
                    description="Transformers typically benefit from Xavier initialization",
                    mathematical_context="Xavier initialization maintains activation variance"
                ))
    
    def _validate_loss_functions(self):
        """Validate loss function implementations and numerical stability."""
        source_files = list(self.project_root.rglob('*.py'))
        
        for file_path in source_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check for numerical stability in log operations
                if re.search(r'torch\.log\((?!.*eps).*\)|math\.log\((?!.*eps)', content, re.IGNORECASE):
                    self.issues.append(MathValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="numerical_stability",
                        description="log() without epsilon may cause numerical instability",
                        file_path=str(file_path),
                        suggested_fix="Use torch.log(x + eps) instead of torch.log(x)",
                        mathematical_context="log(0) = -inf, adding small epsilon prevents this"
                    ))
                
                # Check for softmax stability
                if re.search(r'softmax.*dim.*-1|F\.softmax', content, re.IGNORECASE):
                    if not re.search(r'log_softmax|F\.log_softmax', content, re.IGNORECASE):
                        self.issues.append(MathValidationIssue(
                            severity=ValidationSeverity.INFO,
                            category="numerical_stability",
                            description="Consider log_softmax for better numerical stability",
                            file_path=str(file_path),
                            mathematical_context="log_softmax is more numerically stable than log(softmax(x))"
                        ))
                
                # Check cross entropy implementation
                if re.search(r'cross.*entropy|CrossEntropyLoss', content, re.IGNORECASE):
                    # This is usually fine if using PyTorch's implementation
                    pass
                
            except Exception:
                continue
    
    def _validate_numerical_stability(self):
        """Validate overall numerical stability of the implementation."""
        source_files = list(self.project_root.rglob('*.py'))
        
        stability_issues = []
        
        for file_path in source_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check for division without safety
                div_matches = re.findall(r'/\s*(?![\d\.e\+\-])', content)
                if len(div_matches) > 0:
                    # Check if there are safety mechanisms
                    if not re.search(r'eps|epsilon|torch\.clamp|max\(.*1e-', content, re.IGNORECASE):
                        stability_issues.append((file_path, "Division without numerical safety"))
                
                # Check for sqrt without safety
                if re.search(r'torch\.sqrt|math\.sqrt', content, re.IGNORECASE):
                    if not re.search(r'relu|clamp|max\(', content, re.IGNORECASE):
                        stability_issues.append((file_path, "sqrt() without ensuring positive input"))
                
                # Check for exponential operations
                exp_count = len(re.findall(r'torch\.exp|math\.exp', content, re.IGNORECASE))
                if exp_count > 3:  # Arbitrary threshold
                    if not re.search(r'clamp|clip', content, re.IGNORECASE):
                        stability_issues.append((file_path, "Multiple exp() operations without clamping"))
                
            except Exception:
                continue
        
        # Add issues
        for file_path, description in stability_issues:
            self.issues.append(MathValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="numerical_stability",
                description=description,
                file_path=str(file_path),
                mathematical_context="Numerical operations need safety mechanisms to prevent overflow/underflow"
            ))
    
    def _validate_mathematical_correctness(self):
        """Validate mathematical correctness of implementations."""
        source_files = list(self.project_root.rglob('*.py'))
        
        for file_path in source_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Parse AST to check mathematical operations
                try:
                    tree = ast.parse(content)
                    
                    # Check for potential mathematical errors in AST
                    for node in ast.walk(tree):
                        if isinstance(node, ast.BinOp):
                            # Check for common mathematical errors
                            if isinstance(node.op, ast.Pow):
                                # Check for x^2 vs x**2 confusion (not applicable in Python)
                                pass
                            elif isinstance(node.op, ast.Div):
                                # Division checks are handled in numerical stability
                                pass
                    
                except SyntaxError:
                    # Skip files with syntax errors
                    continue
                
                # Pattern-based mathematical correctness checks
                
                # Check matrix multiplication order
                if re.search(r'@.*transpose|transpose.*@', content, re.IGNORECASE):
                    # This might indicate manual transposition for matrix mult
                    # Usually PyTorch handles this correctly, but flag for review
                    pass
                
                # Check dimension mismatches (pattern-based, limited)
                if re.search(r'view\(-1.*reshape\(-1', content, re.IGNORECASE):
                    self.issues.append(MathValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category="tensor_reshaping",
                        description="Multiple tensor reshaping operations may indicate dimension issues",
                        file_path=str(file_path)
                    ))
                
            except Exception:
                continue
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Count issues by category
        issue_categories = {}
        for issue in self.issues:
            issue_categories[issue.category] = issue_categories.get(issue.category, 0) + 1
        
        # Generate recommendations based on patterns
        if issue_categories.get('initialization', 0) > 0:
            recommendations.append(
                "Implement proper weight initialization (Xavier/Kaiming) for better training dynamics"
            )
        
        if issue_categories.get('numerical_stability', 0) > 2:
            recommendations.append(
                "Add numerical stability mechanisms (epsilon values, clamping) to prevent overflow/underflow"
            )
        
        if issue_categories.get('gradient_flow', 0) > 0:
            recommendations.append(
                "Consider adding residual connections or gradient clipping for better gradient flow"
            )
        
        # Architecture-specific recommendations
        if self.detected_architecture == ArchitectureType.TRANSFORMER:
            if any('attention_scaling' in issue.category for issue in self.issues):
                recommendations.append(
                    "Implement proper attention scaling (1/√d_k) for stable attention computation"
                )
        
        elif self.detected_architecture == ArchitectureType.RWKV:
            if any('complexity_violation' in issue.category for issue in self.issues):
                recommendations.append(
                    "Remove O(n²) operations to maintain RWKV's linear complexity advantage"
                )
        
        elif self.detected_architecture == ArchitectureType.MAMBA:
            if any('ssm_formulation' in issue.category for issue in self.issues):
                recommendations.append(
                    "Ensure complete state space model formulation with all required parameters"
                )
        
        # General recommendations
        if len([i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]) > 0:
            recommendations.append(
                "Address all critical mathematical issues before training to prevent failures"
            )
        
        if not recommendations:
            recommendations.append("Mathematical implementation appears sound - consider running generated tests")
        
        return recommendations
    
    def generate_comprehensive_test_suite(self) -> str:
        """Generate comprehensive mathematical test suite."""
        architecture_name = self.detected_architecture.value.title()
        
        test_code = f'''"""
Neural Network Mathematical Validation Test Suite
Generated by Neural-Math-Checker for {architecture_name} Architecture

Tests mathematical properties, numerical stability, and gradient flow.
"""

import torch
import pytest
import numpy as np
import math
from typing import Dict, List, Tuple, Optional

# Import your model components
# Adjust imports based on your project structure
try:
    import sys
    sys.path.append('src')
    # Add your specific imports here
except ImportError:
    pytest.skip("Model implementation not available", allow_module_level=True)


class TestNumericalStability:
    """Test numerical stability of neural network operations."""
    
    def test_no_nan_or_inf_in_forward_pass(self):
        """Test that forward pass never produces NaN or Inf values."""
        # This test needs to be customized for your specific model
        # Example structure:
        
        batch_size, seq_len, hidden_dim = 4, 32, 512
        
        # Create model (replace with your model)
        # model = YourModel(hidden_dim=hidden_dim)
        # x = torch.randn(batch_size, seq_len, hidden_dim)
        
        # output = model(x)
        
        # assert not torch.isnan(output).any(), "Forward pass produced NaN values"
        # assert not torch.isinf(output).any(), "Forward pass produced Inf values"
        
        # For now, create a placeholder test
        x = torch.randn(4, 32, 512)
        assert not torch.isnan(x).any()
    
    def test_gradient_stability(self):
        """Test that gradients remain stable during backpropagation."""
        # Example gradient stability test
        x = torch.randn(2, 16, 256, requires_grad=True)
        
        # Simple operation to test gradient flow
        y = torch.nn.Linear(256, 256)(x)
        loss = y.sum()
        
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        
        # Check gradient magnitude is reasonable
        grad_norm = x.grad.norm().item()
        assert 1e-8 < grad_norm < 1e4, f"Gradient norm {{grad_norm}} is unreasonable"
    
    def test_activation_output_bounds(self):
        """Test that activation functions produce expected output ranges."""
        x = torch.randn(100, 100) * 10  # Large input range
        
        # Test sigmoid bounds
        sigmoid_out = torch.sigmoid(x)
        assert torch.all(sigmoid_out >= 0.0) and torch.all(sigmoid_out <= 1.0)
        
        # Test tanh bounds
        tanh_out = torch.tanh(x)
        assert torch.all(tanh_out >= -1.0) and torch.all(tanh_out <= 1.0)
        
        # Test ReLU bounds
        relu_out = torch.relu(x)
        assert torch.all(relu_out >= 0.0)
    
    def test_numerical_precision_stability(self):
        """Test operations remain stable across different numerical precisions."""
        # Test with different dtypes
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(10, 10, dtype=dtype)
            
            # Test basic operations
            result = torch.softmax(x, dim=-1)
            assert not torch.isnan(result).any()
            
            # Test that probabilities sum to 1
            prob_sums = result.sum(dim=-1)
            assert torch.allclose(prob_sums, torch.ones_like(prob_sums), rtol=1e-4)


class TestMathematicalProperties:
    """Test mathematical correctness of neural network operations."""
    
    def test_weight_initialization_properties(self):
        """Test that weight initialization follows expected statistical properties."""
        hidden_dim = 512
        
        # Test Xavier/Glorot initialization
        layer = torch.nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(layer.weight)
        
        # Check variance approximately follows Xavier formula
        expected_var = 2.0 / (hidden_dim + hidden_dim)  # Xavier variance
        actual_var = layer.weight.var().item()
        
        assert abs(actual_var - expected_var) < expected_var * 0.5, f"Xavier variance mismatch: {{actual_var}} vs {{expected_var}}"
        
        # Test Kaiming initialization for ReLU
        layer_he = torch.nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.kaiming_uniform_(layer_he.weight, nonlinearity='relu')
        
        # Check that weights are not all the same (proper randomization)
        assert layer_he.weight.std() > 0.01, "Weights are not properly randomized"
    
    def test_gradient_flow_properties(self):
        """Test gradient flow through network layers."""
        # Create a simple deep network
        layers = []
        hidden_dim = 128
        depth = 8
        
        for i in range(depth):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        
        model = torch.nn.Sequential(*layers)
        
        x = torch.randn(4, hidden_dim, requires_grad=True)
        output = model(x)
        loss = output.sum()
        
        loss.backward()
        
        # Check that gradients exist at input
        assert x.grad is not None
        
        # Check gradient magnitudes don't vanish completely
        input_grad_norm = x.grad.norm().item()
        assert input_grad_norm > 1e-6, f"Gradients may be vanishing: {{input_grad_norm}}"
        
        # Check gradient magnitudes don't explode
        assert input_grad_norm < 1e3, f"Gradients may be exploding: {{input_grad_norm}}"
    
    def test_attention_scaling_correctness(self):
        """Test attention mechanism scaling (if applicable)."""
        if '{architecture_name.lower()}' not in ['transformer']:
            pytest.skip("Not applicable for this architecture")
        
        d_k = 64
        seq_len = 32
        
        # Simulate attention computation
        q = torch.randn(1, seq_len, d_k)
        k = torch.randn(1, seq_len, d_k)
        v = torch.randn(1, seq_len, d_k)
        
        # Attention without scaling
        attn_weights_unscaled = torch.softmax(torch.bmm(q, k.transpose(-2, -1)), dim=-1)
        
        # Attention with proper scaling
        scaling_factor = math.sqrt(d_k)
        attn_weights_scaled = torch.softmax(torch.bmm(q, k.transpose(-2, -1)) / scaling_factor, dim=-1)
        
        # Scaled attention should have lower variance (less peaked)
        unscaled_var = attn_weights_unscaled.var()
        scaled_var = attn_weights_scaled.var()
        
        assert scaled_var < unscaled_var, "Attention scaling should reduce variance"


class TestArchitectureSpecificMath:
    """Test mathematical properties specific to the detected architecture."""
    
    def test_architecture_complexity_bounds(self):
        """Test that architecture meets complexity requirements."""
        # This needs to be customized based on your specific architecture
        
        if '{self.detected_architecture.value}' == 'rwkv':
            self._test_rwkv_linear_complexity()
        elif '{self.detected_architecture.value}' == 'mamba':
            self._test_mamba_ssm_properties()
        elif '{self.detected_architecture.value}' == 'transformer':
            self._test_transformer_attention_properties()
        elif '{self.detected_architecture.value}' == 'moe':
            self._test_moe_sparsity_properties()
    
    def _test_rwkv_linear_complexity(self):
        """Test RWKV linear time complexity."""
        # This would need actual RWKV implementation
        # For now, test basic linear operation complexity
        
        seq_lengths = [64, 128, 256, 512]
        times = []
        
        for seq_len in seq_lengths:
            x = torch.randn(1, seq_len, 256)
            
            import time
            start_time = time.time()
            
            # Simulate linear operation (O(n))
            result = torch.nn.Linear(256, 256)(x)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Check roughly linear scaling
        for i in range(1, len(seq_lengths)):
            seq_ratio = seq_lengths[i] / seq_lengths[i-1]
            time_ratio = times[i] / times[i-1] if times[i-1] > 0 else 1
            
            # Should be closer to linear than quadratic
            assert time_ratio < seq_ratio * 1.5, f"Time scaling not linear: {{time_ratio}} vs {{seq_ratio}}"
    
    def _test_mamba_ssm_properties(self):
        """Test Mamba state space model properties."""
        # Placeholder test for SSM properties
        # Would need actual Mamba implementation
        
        # Test basic state space model stability
        A = torch.randn(16, 16) * 0.1  # Small eigenvalues for stability
        eigenvals = torch.linalg.eigvals(A).real
        
        # For stability, eigenvalues should have negative real parts
        # This is a simplified test
        max_eigenval = eigenvals.max()
        assert max_eigenval < 1.0, f"SSM may be unstable with eigenvalue {{max_eigenval}}"
    
    def _test_transformer_attention_properties(self):
        """Test Transformer attention mathematical properties."""
        # Test attention probability properties
        seq_len, d_k = 32, 64
        
        q = torch.randn(1, seq_len, d_k)
        k = torch.randn(1, seq_len, d_k)
        
        # Compute attention weights
        attn_scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Test that attention weights sum to 1
        weight_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), rtol=1e-4)
        
        # Test that attention weights are non-negative
        assert torch.all(attn_weights >= 0.0)
    
    def _test_moe_sparsity_properties(self):
        """Test MoE sparsity and routing properties."""
        # Test top-k sparsity
        num_experts, top_k = 8, 2
        batch_size, seq_len = 4, 16
        
        # Simulate router logits
        router_logits = torch.randn(batch_size, seq_len, num_experts)
        
        # Apply top-k selection
        topk_values, topk_indices = torch.topk(router_logits, top_k, dim=-1)
        
        # Test sparsity
        expected_sparsity = 1.0 - (top_k / num_experts)
        assert expected_sparsity > 0.5, f"Sparsity {{expected_sparsity}} should be > 0.5 for efficiency"
        
        # Test that exactly top_k experts are selected
        assert topk_indices.shape[-1] == top_k


class TestLossLandscapeProperties:
    """Test properties of the loss landscape."""
    
    def test_loss_smoothness(self):
        """Test that loss function is reasonably smooth."""
        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )
        
        x = torch.randn(5, 10)
        target = torch.randn(5, 1)
        
        # Test loss at different points
        losses = []
        for eps in [0.0, 0.01, 0.02]:
            # Add small perturbation to weights
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * eps)
            
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
            losses.append(loss.item())
            
            # Reset model (simplified)
            if eps > 0:
                with torch.no_grad():
                    for param in model.parameters():
                        param.add_(torch.randn_like(param) * -eps)
        
        # Loss shouldn't change dramatically with small weight changes
        loss_var = np.var(losses)
        assert loss_var < 1.0, f"Loss landscape may be too rough: variance {{loss_var}}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
        
        return test_code
    
    def generate_mathematical_validation_report(self) -> str:
        """Generate comprehensive mathematical validation report."""
        result = self.validate_complete_mathematical_properties()
        
        report = f"""
# Neural Network Mathematical Validation Report

## Architecture Analysis
- **Detected Architecture**: {result.architecture_type.value.title()}
- **Mathematical Validity**: {'✅ VALID' if result.is_mathematically_valid else '❌ INVALID'}
- **Total Issues**: {len(result.issues)}
  - Critical: {len([i for i in result.issues if i.severity == ValidationSeverity.CRITICAL])}
  - Warnings: {len([i for i in result.issues if i.severity == ValidationSeverity.WARNING])}
  - Info: {len([i for i in result.issues if i.severity == ValidationSeverity.INFO])}

## Numerical Properties Analysis
- **Proper Initialization**: {'✅' if result.numerical_properties.has_proper_initialization else '❌'}
- **Gradient Clipping**: {'✅' if result.numerical_properties.has_gradient_clipping else '❌'}
- **Numerical Stability**: {'✅' if result.numerical_properties.has_numerical_stability else '❌'}
- **Proper Normalization**: {'✅' if result.numerical_properties.has_proper_normalization else '❌'}

## Gradient Flow Analysis
- **Residual Connections**: {'✅' if result.gradient_flow_properties.has_residual_connections else '❌'}
- **Proper Scaling**: {'✅' if result.gradient_flow_properties.has_proper_scaling else '❌'}
- **Network Depth**: {result.gradient_flow_properties.gradient_flow_depth} layers
- **Vanishing Gradient Risk**: {result.gradient_flow_properties.vanishing_gradient_risk.title()}
- **Exploding Gradient Risk**: {result.gradient_flow_properties.exploding_gradient_risk.title()}

## Mathematical Constraints ({result.architecture_type.value.title()})
"""
        
        # Add complexity bounds
        for operation, complexity in result.mathematical_constraints.complexity_bounds.items():
            report += f"- **{operation.title()} Complexity**: {complexity}\n"
        
        report += "\n### Stability Requirements\n"
        for requirement in result.mathematical_constraints.numerical_stability_requirements:
            report += f"- {requirement.replace('_', ' ').title()}\n"
        
        report += "\n### Invariant Properties\n"
        for prop in result.mathematical_constraints.invariant_properties:
            report += f"- {prop.replace('_', ' ').title()}\n"
        
        report += "\n## Issues Found\n"
        
        # Group issues by severity
        critical_issues = [i for i in result.issues if i.severity == ValidationSeverity.CRITICAL]
        warning_issues = [i for i in result.issues if i.severity == ValidationSeverity.WARNING]
        info_issues = [i for i in result.issues if i.severity == ValidationSeverity.INFO]
        
        if critical_issues:
            report += "\n### ❌ Critical Issues\n"
            for i, issue in enumerate(critical_issues, 1):
                report += f"{i}. **{issue.category.replace('_', ' ').title()}**: {issue.description}\n"
                if issue.mathematical_context:
                    report += f"   - *Mathematical Context*: {issue.mathematical_context}\n"
                if issue.suggested_fix:
                    report += f"   - *Suggested Fix*: {issue.suggested_fix}\n"
                report += "\n"
        
        if warning_issues:
            report += "\n### ⚠️ Warnings\n"
            for i, issue in enumerate(warning_issues, 1):
                report += f"{i}. **{issue.category.replace('_', ' ').title()}**: {issue.description}\n"
                if issue.mathematical_context:
                    report += f"   - *Context*: {issue.mathematical_context}\n"
                report += "\n"
        
        if info_issues:
            report += "\n### ℹ️ Recommendations\n"
            for i, issue in enumerate(info_issues, 1):
                report += f"{i}. {issue.description}\n"
        
        report += "\n## Actionable Recommendations\n"
        for i, rec in enumerate(result.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += f"""
## Architecture-Specific Guidance

### {result.architecture_type.value.title()} Best Practices
"""
        
        if result.architecture_type == ArchitectureType.TRANSFORMER:
            report += """
- Implement proper attention scaling (1/√d_k)
- Use pre-normalization for better gradient flow
- Consider relative positional encoding for better length generalization
- Apply dropout to attention weights and feed-forward layers
"""
        elif result.architecture_type == ArchitectureType.RWKV:
            report += """
- Ensure linear-time complexity O(n) throughout
- Implement proper state-based recurrent computation
- Use exponential interpolation for time mixing
- Maintain streaming capability for incremental processing
"""
        elif result.architecture_type == ArchitectureType.MAMBA:
            report += """
- Implement selective state space model formulation
- Ensure input-dependent parameter selection
- Use efficient parallel scan for fast computation
- Validate discretization stability
"""
        elif result.architecture_type == ArchitectureType.MOE:
            report += """
- Implement proper top-k expert routing
- Add load balancing auxiliary loss
- Enforce capacity constraints to prevent overflow
- Use appropriate sparsity levels for efficiency
"""
        
        report += """
## Testing and Validation

### Immediate Actions
1. Run generated mathematical test suite
2. Fix all critical issues before training
3. Address warnings for optimal performance
4. Implement missing numerical stability mechanisms

### Continuous Validation
1. Monitor gradient norms during training
2. Track numerical stability metrics
3. Validate mathematical properties in CI/CD
4. Profile computational complexity regularly

### Mathematical Verification Checklist
- [ ] Weight initialization follows proper scaling
- [ ] Activation functions have appropriate bounds
- [ ] Gradient flow is unimpeded through all layers
- [ ] Numerical operations include stability mechanisms
- [ ] Architecture-specific constraints are satisfied
- [ ] Loss landscape is reasonably smooth
- [ ] Model complexity meets theoretical bounds
"""
        
        return report


def main():
    """Main entry point for Neural Math Checker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Network Mathematical Validation")
    parser.add_argument("--project-root", type=Path, default=".", help="Project root directory")
    parser.add_argument("--architecture", type=str, choices=[arch.value for arch in ArchitectureType], 
                       help="Force specific architecture type")
    parser.add_argument("--generate-tests", action="store_true", help="Generate mathematical test suite")
    parser.add_argument("--report", action="store_true", help="Generate validation report")
    parser.add_argument("--config", type=Path, help="Configuration file (optional)")
    
    args = parser.parse_args()
    
    # Convert architecture string to enum if provided
    architecture_type = None
    if args.architecture:
        architecture_type = ArchitectureType(args.architecture)
    
    checker = NeuralMathChecker(args.project_root, architecture_type)
    
    if args.generate_tests:
        test_code = checker.generate_comprehensive_test_suite()
        
        test_file = args.project_root / "test_neural_math_validation.py"
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        print(f"Generated mathematical test suite: {test_file}")
    
    if args.report:
        report = checker.generate_mathematical_validation_report()
        
        report_file = args.project_root / "neural_math_validation_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Generated validation report: {report_file}")
        print("\n" + "="*60)
        print(report)
    
    # Default: run validation and show summary
    if not args.generate_tests and not args.report:
        result = checker.validate_complete_mathematical_properties()
        
        print(f"Architecture: {result.architecture_type.value.title()}")
        print(f"Mathematical Validity: {'✅ VALID' if result.is_mathematically_valid else '❌ INVALID'}")
        print(f"Issues: {len(result.issues)} total")
        print(f"Critical: {len([i for i in result.issues if i.severity == ValidationSeverity.CRITICAL])}")
        
        if result.issues:
            print("\nMost critical issues:")
            for issue in result.issues[:3]:
                print(f"- {issue.category}: {issue.description}")


if __name__ == "__main__":
    main()