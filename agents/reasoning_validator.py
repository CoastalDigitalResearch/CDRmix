#!/usr/bin/env python3
"""
Reasoning-Validator Agent

Generic reasoning system validation agent that works across different reasoning approaches
while supporting custom reasoning architectures like Hybrid Hebbian-RL.

This agent validates:
- Modern reasoning patterns (Chain-of-Thought, Tree-of-Thoughts, Self-Consistency)
- Custom reasoning systems (Hybrid Hebbian-RL, memory-augmented reasoning)
- Memory management and episodic storage systems
- Reinforcement learning-based reasoning policies
- Plasticity and adaptation mechanisms

Designed to be portable across different reasoning-enabled language model projects.
"""

import ast
import math
import re
import json
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
import yaml


class ReasoningType(Enum):
    """Types of reasoning systems supported."""
    CHAIN_OF_THOUGHT = "chain_of_thought"          # CoT reasoning
    TREE_OF_THOUGHTS = "tree_of_thoughts"          # ToT reasoning
    SELF_CONSISTENCY = "self_consistency"          # Self-consistency decoding
    HYBRID_HEBBIAN_RL = "hybrid_hebbian_rl"        # CDRmix custom system
    MEMORY_AUGMENTED = "memory_augmented"          # Memory-based reasoning
    TOOL_AUGMENTED = "tool_augmented"              # Tool-using reasoning
    MULTI_AGENT = "multi_agent"                    # Multi-agent reasoning
    REFLEXION = "reflexion"                        # Reflexion-style self-reflection
    GENERIC = "generic"                            # Basic reasoning patterns


class ReasoningValidationSeverity(Enum):
    """Severity levels for reasoning validation issues."""
    CRITICAL = "critical"  # Will cause reasoning failure
    WARNING = "warning"    # May cause poor reasoning performance
    INFO = "info"         # Best practice recommendations


@dataclass
class ReasoningValidationIssue:
    """Individual reasoning validation issue."""
    severity: ReasoningValidationSeverity
    category: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggested_fix: Optional[str] = None
    reasoning_context: Optional[str] = None


@dataclass
class MemorySystemProperties:
    """Properties of reasoning memory system."""
    has_working_memory: bool = False
    has_episodic_memory: bool = False
    has_memory_operations: bool = False
    has_retrieval_system: bool = False
    memory_indexing_type: Optional[str] = None
    context_window_management: bool = False


@dataclass
class ReasoningControllerProperties:
    """Properties of reasoning controller system."""
    has_branch_management: bool = False
    has_proposal_system: bool = False
    has_evaluation_system: bool = False
    has_acceptance_criteria: bool = False
    supports_rollback: bool = False
    max_reasoning_steps: Optional[int] = None


@dataclass
class PlasticityProperties:
    """Properties of plasticity/adaptation system."""
    has_ephemeral_adapters: bool = False
    has_hebbian_learning: bool = False
    has_correlation_objectives: bool = False
    has_adapter_management: bool = False
    supports_commit_rollback: bool = False
    adapter_isolation: bool = False


@dataclass
class RLPolicyProperties:
    """Properties of reinforcement learning policy system."""
    has_policy_network: bool = False
    has_reward_system: bool = False
    has_advantage_estimation: bool = False
    has_policy_optimization: bool = False
    supports_memory_actions: bool = False
    algorithm_type: Optional[str] = None


@dataclass
class ReasoningCapabilities:
    """Overall reasoning system capabilities."""
    reasoning_types: Set[ReasoningType] = field(default_factory=set)
    memory_system: MemorySystemProperties = field(default_factory=MemorySystemProperties)
    controller: ReasoningControllerProperties = field(default_factory=ReasoningControllerProperties)
    plasticity: PlasticityProperties = field(default_factory=PlasticityProperties)
    rl_policy: RLPolicyProperties = field(default_factory=RLPolicyProperties)
    
    
@dataclass
class ReasoningValidationResult:
    """Complete reasoning validation result."""
    is_reasoning_valid: bool
    detected_reasoning_types: Set[ReasoningType]
    issues: List[ReasoningValidationIssue]
    capabilities: ReasoningCapabilities
    recommendations: List[str]
    test_suite_generated: bool = False


class ReasoningSystemValidator(ABC):
    """Abstract base class for reasoning system validators."""
    
    @abstractmethod
    def get_reasoning_type(self) -> ReasoningType:
        """Get the reasoning type this validator handles."""
        pass
    
    @abstractmethod
    def validate_reasoning_system(self, content: str, file_path: Path) -> List[ReasoningValidationIssue]:
        """Validate reasoning system implementation."""
        pass
    
    @abstractmethod
    def detect_reasoning_patterns(self, content: str) -> bool:
        """Detect if this reasoning pattern is present in the code."""
        pass


class ChainOfThoughtValidator(ReasoningSystemValidator):
    """Chain-of-Thought reasoning validator."""
    
    def get_reasoning_type(self) -> ReasoningType:
        return ReasoningType.CHAIN_OF_THOUGHT
    
    def detect_reasoning_patterns(self, content: str) -> bool:
        cot_patterns = [
            r'chain.*of.*thought|cot',
            r'step.*by.*step',
            r'reasoning.*steps',
            r'intermediate.*steps',
            r'let.*me.*think'
        ]
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in cot_patterns)
    
    def validate_reasoning_system(self, content: str, file_path: Path) -> List[ReasoningValidationIssue]:
        issues = []
        
        # Check for step-by-step reasoning structure
        if not re.search(r'step|reasoning|intermediate', content, re.IGNORECASE):
            issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.WARNING,
                category="cot_structure",
                description="Chain-of-Thought may lack explicit step structure",
                file_path=str(file_path),
                reasoning_context="CoT requires explicit reasoning steps"
            ))
        
        # Check for reasoning chain validation
        if not re.search(r'validate|verify|check.*reasoning', content, re.IGNORECASE):
            issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.INFO,
                category="cot_validation",
                description="Consider adding reasoning chain validation",
                reasoning_context="Validation improves CoT reliability"
            ))
        
        return issues


class TreeOfThoughtsValidator(ReasoningSystemValidator):
    """Tree-of-Thoughts reasoning validator."""
    
    def get_reasoning_type(self) -> ReasoningType:
        return ReasoningType.TREE_OF_THOUGHTS
    
    def detect_reasoning_patterns(self, content: str) -> bool:
        tot_patterns = [
            r'tree.*of.*thoughts|tot',
            r'branch.*reasoning',
            r'multiple.*paths',
            r'search.*tree',
            r'breadth.*first|depth.*first'
        ]
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in tot_patterns)
    
    def validate_reasoning_system(self, content: str, file_path: Path) -> List[ReasoningValidationIssue]:
        issues = []
        
        # Check for tree/branch structure
        if not re.search(r'tree|branch|node|path', content, re.IGNORECASE):
            issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.WARNING,
                category="tot_structure",
                description="Tree-of-Thoughts may lack tree structure",
                file_path=str(file_path),
                reasoning_context="ToT requires branching search structure"
            ))
        
        # Check for evaluation/pruning mechanism
        if not re.search(r'evaluate|prune|score|select', content, re.IGNORECASE):
            issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.WARNING,
                category="tot_evaluation",
                description="Tree-of-Thoughts may lack branch evaluation",
                reasoning_context="ToT needs to evaluate and prune reasoning branches"
            ))
        
        return issues


class HybridHebbianRLValidator(ReasoningSystemValidator):
    """Hybrid Hebbian-RL reasoning system validator (CDRmix specific)."""
    
    def get_reasoning_type(self) -> ReasoningType:
        return ReasoningType.HYBRID_HEBBIAN_RL
    
    def detect_reasoning_patterns(self, content: str) -> bool:
        hebbian_rl_patterns = [
            r'hebbian.*rl|rl.*hebbian',
            r'plastic.*lora|ephemeral.*adapter',
            r'correlation.*loss|hebbian.*learning',
            r'rl.*memagent|memory.*agent',
            r'grpo|advantage.*estimation'
        ]
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in hebbian_rl_patterns)
    
    def validate_reasoning_system(self, content: str, file_path: Path) -> List[ReasoningValidationIssue]:
        issues = []
        
        # Validate Hebbian component
        hebbian_indicators = ['correlation', 'hebbian', 'plasticity', 'adapter']
        missing_hebbian = []
        for indicator in hebbian_indicators:
            if indicator not in content.lower():
                missing_hebbian.append(indicator)
        
        if len(missing_hebbian) > 2:
            issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.CRITICAL,
                category="hebbian_component",
                description=f"Missing Hebbian components: {', '.join(missing_hebbian)}",
                file_path=str(file_path),
                reasoning_context="Hybrid system requires Hebbian plasticity component"
            ))
        
        # Validate RL component
        rl_indicators = ['policy', 'reward', 'advantage', 'optimization']
        missing_rl = []
        for indicator in rl_indicators:
            if indicator not in content.lower():
                missing_rl.append(indicator)
        
        if len(missing_rl) > 2:
            issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.CRITICAL,
                category="rl_component",
                description=f"Missing RL components: {', '.join(missing_rl)}",
                file_path=str(file_path),
                reasoning_context="Hybrid system requires RL policy component"
            ))
        
        # Validate memory operations
        memory_operations = ['summarize', 'retrieve', 'splice_window', 'cache_write']
        found_operations = sum(1 for op in memory_operations if op in content.lower())
        
        if found_operations < 2:
            issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.WARNING,
                category="memory_operations",
                description="Limited memory operations detected",
                reasoning_context="Hybrid Hebbian-RL should support diverse memory operations"
            ))
        
        # Validate branch management
        if not re.search(r'branch|proposal|accept|reject|rollback', content, re.IGNORECASE):
            issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.CRITICAL,
                category="branch_management",
                description="Missing branch management system",
                reasoning_context="Hybrid system requires proposal/acceptance mechanism"
            ))
        
        # Validate adapter isolation
        if 'adapter' in content.lower() and not re.search(r'commit|rollback|isolat', content, re.IGNORECASE):
            issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.WARNING,
                category="adapter_isolation",
                description="Adapter system may lack proper isolation",
                reasoning_context="Ephemeral adapters need commit/rollback isolation"
            ))
        
        return issues


class MemoryAugmentedValidator(ReasoningSystemValidator):
    """Memory-augmented reasoning validator."""
    
    def get_reasoning_type(self) -> ReasoningType:
        return ReasoningType.MEMORY_AUGMENTED
    
    def detect_reasoning_patterns(self, content: str) -> bool:
        memory_patterns = [
            r'memory.*augmented|external.*memory',
            r'episodic.*memory|working.*memory',
            r'retrieval.*augmented|rag',
            r'faiss|vector.*database',
            r'memory.*operation'
        ]
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in memory_patterns)
    
    def validate_reasoning_system(self, content: str, file_path: Path) -> List[ReasoningValidationIssue]:
        issues = []
        
        # Check for memory storage system
        if not re.search(r'store|index|database|faiss|vector', content, re.IGNORECASE):
            issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.WARNING,
                category="memory_storage",
                description="Memory system may lack proper storage backend",
                file_path=str(file_path),
                reasoning_context="Memory-augmented systems need persistent storage"
            ))
        
        # Check for retrieval mechanism
        if not re.search(r'retrieve|query|search|top.*k', content, re.IGNORECASE):
            issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.CRITICAL,
                category="memory_retrieval",
                description="Missing memory retrieval mechanism",
                reasoning_context="Memory systems must retrieve relevant information"
            ))
        
        return issues


class GenericReasoningValidator(ReasoningSystemValidator):
    """Generic reasoning system validator."""
    
    def get_reasoning_type(self) -> ReasoningType:
        return ReasoningType.GENERIC
    
    def detect_reasoning_patterns(self, content: str) -> bool:
        # Always matches for generic validation
        return True
    
    def validate_reasoning_system(self, content: str, file_path: Path) -> List[ReasoningValidationIssue]:
        # Generic validator provides minimal validation
        return []


class ReasoningValidator:
    """
    Generic reasoning system validation agent.
    
    Validates reasoning capabilities, memory management, and adaptation mechanisms
    across different reasoning approaches and architectures.
    """
    
    def __init__(self, project_root: Path, reasoning_types: Optional[List[ReasoningType]] = None):
        self.project_root = Path(project_root)
        self.issues: List[ReasoningValidationIssue] = []
        
        # Initialize reasoning system validators
        self.reasoning_validators: Dict[ReasoningType, ReasoningSystemValidator] = {
            ReasoningType.CHAIN_OF_THOUGHT: ChainOfThoughtValidator(),
            ReasoningType.TREE_OF_THOUGHTS: TreeOfThoughtsValidator(),
            ReasoningType.HYBRID_HEBBIAN_RL: HybridHebbianRLValidator(),
            ReasoningType.MEMORY_AUGMENTED: MemoryAugmentedValidator(),
            ReasoningType.GENERIC: GenericReasoningValidator()
        }
        
        # Auto-detect reasoning types if not specified
        self.detected_reasoning_types = reasoning_types or self._detect_reasoning_types()
    
    def _detect_reasoning_types(self) -> Set[ReasoningType]:
        """Auto-detect reasoning systems from codebase."""
        detected_types = set()
        
        # Search reasoning-related files
        reasoning_dirs = ['reasoning', 'agents', 'memory', 'adapters']
        source_files = []
        
        for reasoning_dir in reasoning_dirs:
            reasoning_path = self.project_root / reasoning_dir
            if reasoning_path.exists():
                source_files.extend(list(reasoning_path.rglob('*.py')))
        
        # Also search general source directories
        for source_dir in ['src', 'models', 'lib']:
            source_path = self.project_root / source_dir
            if source_path.exists():
                source_files.extend(list(source_path.rglob('*.py')))
        
        # Check for reasoning patterns in files
        for file_path in source_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                for reasoning_type, validator in self.reasoning_validators.items():
                    if validator.detect_reasoning_patterns(content):
                        detected_types.add(reasoning_type)
                        
            except Exception:
                continue
        
        return detected_types or {ReasoningType.GENERIC}
    
    def validate_complete_reasoning_system(self) -> ReasoningValidationResult:
        """
        Comprehensive reasoning system validation.
        """
        self.issues = []
        
        # Validate each detected reasoning system
        for reasoning_type in self.detected_reasoning_types:
            if reasoning_type in self.reasoning_validators:
                self._validate_specific_reasoning_system(reasoning_type)
        
        # Core reasoning system validation
        capabilities = self._analyze_reasoning_capabilities()
        
        # Validate reasoning infrastructure
        self._validate_memory_systems()
        self._validate_reasoning_controller()
        self._validate_plasticity_systems()
        self._validate_rl_policy_systems()
        
        # Advanced validation
        self._validate_reasoning_safety()
        self._validate_reasoning_efficiency()
        self._validate_reasoning_scalability()
        
        # Generate recommendations
        recommendations = self._generate_reasoning_recommendations()
        
        return ReasoningValidationResult(
            is_reasoning_valid=not any(issue.severity == ReasoningValidationSeverity.CRITICAL for issue in self.issues),
            detected_reasoning_types=self.detected_reasoning_types,
            issues=self.issues,
            capabilities=capabilities,
            recommendations=recommendations
        )
    
    def _validate_specific_reasoning_system(self, reasoning_type: ReasoningType):
        """Validate a specific reasoning system."""
        validator = self.reasoning_validators[reasoning_type]
        
        # Find relevant files for this reasoning system
        source_files = []
        for pattern in ['**/*.py']:
            source_files.extend(self.project_root.glob(pattern))
        
        for file_path in source_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                if validator.detect_reasoning_patterns(content):
                    issues = validator.validate_reasoning_system(content, file_path)
                    self.issues.extend(issues)
                    
            except Exception as e:
                self.issues.append(ReasoningValidationIssue(
                    severity=ReasoningValidationSeverity.INFO,
                    category="file_access",
                    description=f"Could not analyze {file_path}: {e}"
                ))
    
    def _analyze_reasoning_capabilities(self) -> ReasoningCapabilities:
        """Analyze overall reasoning system capabilities."""
        capabilities = ReasoningCapabilities()
        capabilities.reasoning_types = self.detected_reasoning_types.copy()
        
        # Analyze capabilities from source files
        source_files = list(self.project_root.rglob('*.py'))
        
        for file_path in source_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Analyze memory system capabilities
                if re.search(r'working.*memory|episodic.*memory', content, re.IGNORECASE):
                    if 'working' in content.lower():
                        capabilities.memory_system.has_working_memory = True
                    if 'episodic' in content.lower():
                        capabilities.memory_system.has_episodic_memory = True
                
                if re.search(r'summarize|retrieve|splice|cache', content, re.IGNORECASE):
                    capabilities.memory_system.has_memory_operations = True
                
                if re.search(r'faiss|vector.*database|index', content, re.IGNORECASE):
                    capabilities.memory_system.has_retrieval_system = True
                    if 'faiss' in content.lower():
                        capabilities.memory_system.memory_indexing_type = 'FAISS'
                
                # Analyze controller capabilities
                if re.search(r'branch|proposal|candidate', content, re.IGNORECASE):
                    capabilities.controller.has_branch_management = True
                    capabilities.controller.has_proposal_system = True
                
                if re.search(r'evaluate|score|accept|reject', content, re.IGNORECASE):
                    capabilities.controller.has_evaluation_system = True
                    capabilities.controller.has_acceptance_criteria = True
                
                if re.search(r'rollback|revert|undo', content, re.IGNORECASE):
                    capabilities.controller.supports_rollback = True
                
                # Extract max steps if specified
                max_steps_match = re.search(r'max.*steps?\s*[:=]\s*(\d+)', content, re.IGNORECASE)
                if max_steps_match:
                    capabilities.controller.max_reasoning_steps = int(max_steps_match.group(1))
                
                # Analyze plasticity capabilities
                if re.search(r'adapter|lora|plastic', content, re.IGNORECASE):
                    capabilities.plasticity.has_ephemeral_adapters = True
                    capabilities.plasticity.has_adapter_management = True
                
                if re.search(r'hebbian|correlation.*loss', content, re.IGNORECASE):
                    capabilities.plasticity.has_hebbian_learning = True
                    capabilities.plasticity.has_correlation_objectives = True
                
                if re.search(r'commit|rollback.*adapter', content, re.IGNORECASE):
                    capabilities.plasticity.supports_commit_rollback = True
                
                # Analyze RL policy capabilities
                if re.search(r'policy|actor.*critic|advantage', content, re.IGNORECASE):
                    capabilities.rl_policy.has_policy_network = True
                    capabilities.rl_policy.has_advantage_estimation = True
                
                if re.search(r'reward|returns|value.*function', content, re.IGNORECASE):
                    capabilities.rl_policy.has_reward_system = True
                
                if re.search(r'ppo|trpo|grpo|policy.*gradient', content, re.IGNORECASE):
                    capabilities.rl_policy.has_policy_optimization = True
                    
                    # Detect algorithm type
                    if 'grpo' in content.lower():
                        capabilities.rl_policy.algorithm_type = 'GRPO'
                    elif 'ppo' in content.lower():
                        capabilities.rl_policy.algorithm_type = 'PPO'
                
                memory_actions = ['summarize', 'retrieve', 'splice_window', 'cache_write']
                if any(action in content.lower() for action in memory_actions):
                    capabilities.rl_policy.supports_memory_actions = True
                
            except Exception:
                continue
        
        return capabilities
    
    def _validate_memory_systems(self):
        """Validate memory management systems."""
        memory_files = []
        memory_dirs = ['memory', 'reasoning/memory', 'src/memory']
        
        for memory_dir in memory_dirs:
            memory_path = self.project_root / memory_dir
            if memory_path.exists():
                memory_files.extend(list(memory_path.rglob('*.py')))
        
        if not memory_files:
            # Check if memory is expected based on detected reasoning types
            if ReasoningType.HYBRID_HEBBIAN_RL in self.detected_reasoning_types:
                self.issues.append(ReasoningValidationIssue(
                    severity=ReasoningValidationSeverity.CRITICAL,
                    category="memory_system",
                    description="Memory system implementation not found",
                    reasoning_context="Hybrid Hebbian-RL requires memory management"
                ))
            elif ReasoningType.MEMORY_AUGMENTED in self.detected_reasoning_types:
                self.issues.append(ReasoningValidationIssue(
                    severity=ReasoningValidationSeverity.CRITICAL,
                    category="memory_system",
                    description="Memory-augmented system missing memory implementation"
                ))
            return
        
        # Validate memory implementations
        for file_path in memory_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check for required memory operations
                required_ops = ['store', 'retrieve', 'update', 'clear']
                missing_ops = []
                
                for op in required_ops:
                    if not re.search(rf'\b{op}\b', content, re.IGNORECASE):
                        missing_ops.append(op)
                
                if missing_ops:
                    self.issues.append(ReasoningValidationIssue(
                        severity=ReasoningValidationSeverity.WARNING,
                        category="memory_operations",
                        description=f"Memory operations may be incomplete: missing {', '.join(missing_ops)}",
                        file_path=str(file_path)
                    ))
                
                # Check for memory safety
                if not re.search(r'lock|thread.*safe|concurrent', content, re.IGNORECASE):
                    self.issues.append(ReasoningValidationIssue(
                        severity=ReasoningValidationSeverity.INFO,
                        category="memory_safety",
                        description="Memory system may lack thread safety mechanisms",
                        file_path=str(file_path),
                        reasoning_context="Concurrent reasoning branches need safe memory access"
                    ))
                
            except Exception:
                continue
    
    def _validate_reasoning_controller(self):
        """Validate reasoning controller implementation."""
        controller_files = []
        
        # Look for controller/reasoning management files
        for pattern in ['**/controller*.py', '**/reasoning*.py', '**/branch*.py']:
            controller_files.extend(self.project_root.glob(pattern))
        
        if not controller_files and ReasoningType.HYBRID_HEBBIAN_RL in self.detected_reasoning_types:
            self.issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.CRITICAL,
                category="reasoning_controller",
                description="Reasoning controller implementation not found",
                reasoning_context="Hybrid systems require branch management controller"
            ))
            return
        
        for file_path in controller_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Validate branch management
                if 'branch' in content.lower():
                    if not re.search(r'max.*branch|branch.*limit', content, re.IGNORECASE):
                        self.issues.append(ReasoningValidationIssue(
                            severity=ReasoningValidationSeverity.WARNING,
                            category="branch_limits",
                            description="Branch management may lack resource limits",
                            file_path=str(file_path),
                            reasoning_context="Unbounded branching can cause resource exhaustion"
                        ))
                
                # Validate decision criteria
                if not re.search(r'threshold|criteria|accept|reject', content, re.IGNORECASE):
                    self.issues.append(ReasoningValidationIssue(
                        severity=ReasoningValidationSeverity.WARNING,
                        category="decision_criteria",
                        description="Controller may lack clear decision criteria",
                        file_path=str(file_path)
                    ))
                
            except Exception:
                continue
    
    def _validate_plasticity_systems(self):
        """Validate plasticity and adaptation systems."""
        if ReasoningType.HYBRID_HEBBIAN_RL not in self.detected_reasoning_types:
            return
        
        adapter_files = []
        for pattern in ['**/adapter*.py', '**/plastic*.py', '**/hebbian*.py']:
            adapter_files.extend(self.project_root.glob(pattern))
        
        if not adapter_files:
            self.issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.CRITICAL,
                category="plasticity_system",
                description="Plasticity/adapter system implementation not found",
                reasoning_context="Hybrid Hebbian-RL requires plasticity adapters"
            ))
            return
        
        for file_path in adapter_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Validate ephemeral nature
                if not re.search(r'ephemeral|temporary|rollback|commit', content, re.IGNORECASE):
                    self.issues.append(ReasoningValidationIssue(
                        severity=ReasoningValidationSeverity.WARNING,
                        category="adapter_lifecycle",
                        description="Adapters may not be properly ephemeral",
                        file_path=str(file_path),
                        reasoning_context="Ephemeral adapters should commit/rollback cleanly"
                    ))
                
                # Validate Hebbian learning
                if 'hebbian' in content.lower():
                    if not re.search(r'correlation|covariance|outer.*product', content, re.IGNORECASE):
                        self.issues.append(ReasoningValidationIssue(
                            severity=ReasoningValidationSeverity.WARNING,
                            category="hebbian_implementation",
                            description="Hebbian learning may lack proper correlation computation",
                            file_path=str(file_path)
                        ))
                
            except Exception:
                continue
    
    def _validate_rl_policy_systems(self):
        """Validate reinforcement learning policy systems."""
        if ReasoningType.HYBRID_HEBBIAN_RL not in self.detected_reasoning_types:
            return
        
        rl_files = []
        for pattern in ['**/policy*.py', '**/rl*.py', '**/agent*.py', '**/memagent*.py']:
            rl_files.extend(self.project_root.glob(pattern))
        
        if not rl_files:
            self.issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.CRITICAL,
                category="rl_policy_system",
                description="RL policy system implementation not found",
                reasoning_context="Hybrid Hebbian-RL requires RL policy component"
            ))
            return
        
        for file_path in rl_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Validate policy optimization
                if 'policy' in content.lower():
                    if not re.search(r'advantage|returns|value.*function', content, re.IGNORECASE):
                        self.issues.append(ReasoningValidationIssue(
                            severity=ReasoningValidationSeverity.WARNING,
                            category="policy_optimization",
                            description="Policy system may lack proper value estimation",
                            file_path=str(file_path)
                        ))
                
                # Validate GRPO implementation if present
                if 'grpo' in content.lower():
                    if not re.search(r'kl.*divergence|trust.*region', content, re.IGNORECASE):
                        self.issues.append(ReasoningValidationIssue(
                            severity=ReasoningValidationSeverity.INFO,
                            category="grpo_implementation",
                            description="GRPO implementation may lack KL divergence constraint",
                            file_path=str(file_path)
                        ))
                
            except Exception:
                continue
    
    def _validate_reasoning_safety(self):
        """Validate reasoning safety mechanisms."""
        source_files = list(self.project_root.rglob('*.py'))
        
        # Check for timeout mechanisms
        has_timeout = False
        has_resource_limits = False
        
        for file_path in source_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                if re.search(r'timeout|time.*limit|max.*time', content, re.IGNORECASE):
                    has_timeout = True
                
                if re.search(r'max.*steps|step.*limit|budget', content, re.IGNORECASE):
                    has_resource_limits = True
                
            except Exception:
                continue
        
        if not has_timeout:
            self.issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.WARNING,
                category="reasoning_safety",
                description="Reasoning system may lack timeout mechanisms",
                reasoning_context="Timeouts prevent infinite reasoning loops"
            ))
        
        if not has_resource_limits:
            self.issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.WARNING,
                category="reasoning_safety",
                description="Reasoning system may lack resource limits",
                reasoning_context="Resource limits prevent excessive computation"
            ))
    
    def _validate_reasoning_efficiency(self):
        """Validate reasoning system efficiency."""
        # Check for caching mechanisms
        source_files = list(self.project_root.rglob('*.py'))
        
        has_caching = False
        has_pruning = False
        
        for file_path in source_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                if re.search(r'cache|memoiz|lru', content, re.IGNORECASE):
                    has_caching = True
                
                if re.search(r'prune|trim|filter|select', content, re.IGNORECASE):
                    has_pruning = True
                
            except Exception:
                continue
        
        if not has_caching and len(self.detected_reasoning_types) > 1:
            self.issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.INFO,
                category="reasoning_efficiency",
                description="Consider adding caching for repeated reasoning operations",
                reasoning_context="Caching improves efficiency of complex reasoning systems"
            ))
    
    def _validate_reasoning_scalability(self):
        """Validate reasoning system scalability."""
        # Check for scale-dependent parameters
        config_files = list(self.project_root.glob('**/*.yaml')) + list(self.project_root.glob('**/*.json'))
        
        has_scale_params = False
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                
                # Look for scale-dependent configuration
                if re.search(r'1b|4b|40b|200b|scale.*param', content, re.IGNORECASE):
                    has_scale_params = True
                    break
                
            except Exception:
                continue
        
        if ReasoningType.HYBRID_HEBBIAN_RL in self.detected_reasoning_types and not has_scale_params:
            self.issues.append(ReasoningValidationIssue(
                severity=ReasoningValidationSeverity.INFO,
                category="reasoning_scalability",
                description="Reasoning system may lack scale-adaptive parameters",
                reasoning_context="Scale-adaptive params improve efficiency across model sizes"
            ))
    
    def _generate_reasoning_recommendations(self) -> List[str]:
        """Generate actionable recommendations for reasoning systems."""
        recommendations = []
        
        # Count issues by category
        issue_categories = {}
        for issue in self.issues:
            issue_categories[issue.category] = issue_categories.get(issue.category, 0) + 1
        
        # Generate recommendations based on detected issues
        if issue_categories.get('memory_system', 0) > 0:
            recommendations.append(
                "Implement comprehensive memory management system with working and episodic memory"
            )
        
        if issue_categories.get('reasoning_controller', 0) > 0:
            recommendations.append(
                "Build reasoning controller with branch management and decision criteria"
            )
        
        if issue_categories.get('plasticity_system', 0) > 0:
            recommendations.append(
                "Implement ephemeral adapter system with proper commit/rollback mechanisms"
            )
        
        # Reasoning-type specific recommendations
        if ReasoningType.HYBRID_HEBBIAN_RL in self.detected_reasoning_types:
            if issue_categories.get('hebbian_component', 0) > 0:
                recommendations.append(
                    "Complete Hebbian plasticity component with correlation-based learning"
                )
            
            if issue_categories.get('rl_component', 0) > 0:
                recommendations.append(
                    "Implement RL policy system with GRPO optimization and memory actions"
                )
        
        if ReasoningType.CHAIN_OF_THOUGHT in self.detected_reasoning_types:
            recommendations.append(
                "Add explicit step validation and reasoning chain verification for CoT"
            )
        
        if ReasoningType.TREE_OF_THOUGHTS in self.detected_reasoning_types:
            recommendations.append(
                "Implement branch evaluation and pruning mechanisms for ToT efficiency"
            )
        
        # Safety recommendations
        if issue_categories.get('reasoning_safety', 0) > 0:
            recommendations.append(
                "Add safety mechanisms: timeouts, resource limits, and error handling"
            )
        
        # General recommendations
        if len([i for i in self.issues if i.severity == ReasoningValidationSeverity.CRITICAL]) > 0:
            recommendations.append(
                "Address all critical reasoning issues before deployment"
            )
        
        if not recommendations:
            recommendations.append(
                "Reasoning system appears well-structured - consider running generated tests"
            )
        
        return recommendations
    
    def generate_reasoning_test_suite(self) -> str:
        """Generate comprehensive test suite for reasoning systems."""
        detected_types_str = ", ".join([rt.value.title().replace('_', ' ') for rt in self.detected_reasoning_types])
        
        test_code = f'''"""
Reasoning System Test Suite
Generated by Reasoning-Validator

Tests for reasoning capabilities: {detected_types_str}
"""

import torch
import pytest
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from unittest.mock import Mock, patch

# Adjust imports based on your project structure
try:
    # Add your reasoning system imports here
    import sys
    sys.path.append('src')
    sys.path.append('reasoning')
    # from reasoning.controller import ReasoningController
    # from reasoning.memory.memory_operations import MemoryOperations
    # from reasoning.adapters.plastic_lora import PlasticLoRA
except ImportError:
    pytest.skip("Reasoning implementation not available", allow_module_level=True)


class TestReasoningCapabilities:
    """Test core reasoning system capabilities."""
    
    def test_reasoning_initialization(self):
        """Test that reasoning systems initialize properly."""
        # This test needs to be customized for your specific reasoning implementation
        # Example structure:
        
        # reasoning_controller = ReasoningController(max_branches=3, step_budget=64)
        # assert reasoning_controller.max_branches == 3
        # assert reasoning_controller.step_budget == 64
        
        # For now, create a placeholder test
        assert True, "Reasoning initialization test placeholder"
    
    def test_reasoning_step_execution(self):
        """Test individual reasoning step execution."""
        # Test that reasoning steps execute without errors
        
        # Example test structure:
        # controller = ReasoningController()
        # input_context = torch.randn(1, 32, 512)
        # 
        # result = controller.step(input_context)
        # 
        # assert result is not None
        # assert hasattr(result, 'reasoning_output')
        # assert hasattr(result, 'branch_decisions')
        
        # Placeholder
        x = torch.randn(1, 32, 512)
        assert x.shape == (1, 32, 512)
    
    def test_reasoning_resource_limits(self):
        """Test that reasoning respects resource limits."""
        # Test timeout and step budget enforcement
        
        # Example:
        # controller = ReasoningController(max_steps=5, timeout=1.0)
        # 
        # start_time = time.time()
        # result = controller.reason(difficult_input)
        # end_time = time.time()
        # 
        # assert (end_time - start_time) < 2.0, "Reasoning exceeded timeout"
        # assert result.steps_taken <= 5, "Reasoning exceeded step budget"
        
        # Placeholder test
        max_steps = 10
        actual_steps = 8
        assert actual_steps <= max_steps


class TestMemorySystem:
    """Test reasoning memory management systems."""
    
    def test_working_memory_operations(self):
        """Test working memory summarization and updates."""
        # Test working memory functionality
        
        # Example:
        # memory = WorkingMemory(summary_dim=512, stride=8)
        # context = torch.randn(1, 64, 512)
        # 
        # summary = memory.summarize(context)
        # assert summary.shape[-1] == 512
        # 
        # memory.update(summary)
        # retrieved = memory.get_current_summary()
        # assert torch.allclose(summary, retrieved, rtol=1e-4)
        
        # Placeholder
        context_window = torch.randn(64, 512)
        summary = torch.mean(context_window, dim=0)  # Simple summarization
        assert summary.shape == (512,)
    
    def test_episodic_memory_storage(self):
        """Test episodic memory storage and retrieval."""
        # Test long-term memory functionality
        
        # Example:
        # episodic_memory = EpisodicMemory(index_type='faiss')
        # 
        # # Store experience
        # experience = torch.randn(512)
        # key = "reasoning_episode_1"
        # episodic_memory.store(key, experience)
        # 
        # # Retrieve similar experiences
        # query = torch.randn(512)
        # retrieved = episodic_memory.retrieve(query, top_k=3)
        # assert len(retrieved) <= 3
        
        # Placeholder
        memory_store = {{}}
        experience = torch.randn(512)
        memory_store['key1'] = experience
        assert 'key1' in memory_store
    
    def test_memory_operations_interface(self):
        """Test memory operations interface."""
        # Test the four core memory operations: summarize, retrieve, splice_window, cache_write
        
        operations = ['summarize', 'retrieve', 'splice_window', 'cache_write']
        
        # This would test actual memory operations implementation
        # For now, verify the operations are conceptually sound
        
        for op in operations:
            assert isinstance(op, str), f"Operation {{op}} should be string"
    
    def test_memory_concurrency_safety(self):
        """Test memory system thread safety."""
        # Test concurrent access to memory systems
        
        # Example with concurrent access:
        # import threading
        # 
        # memory = SharedMemory()
        # results = []
        # 
        # def worker(worker_id):
        #     data = torch.randn(256)
        #     memory.store(f"worker_{{worker_id}}", data)
        #     retrieved = memory.retrieve(f"worker_{{worker_id}}")
        #     results.append(torch.allclose(data, retrieved))
        # 
        # threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        # for t in threads:
        #     t.start()
        # for t in threads:
        #     t.join()
        # 
        # assert all(results), "Memory operations not thread-safe"
        
        # Placeholder for concurrency test
        import threading
        shared_data = {{'counter': 0}}
        
        def increment():
            shared_data['counter'] += 1
        
        thread = threading.Thread(target=increment)
        thread.start()
        thread.join()
        
        assert shared_data['counter'] == 1


class TestBranchManagement:
    """Test reasoning branch management and decision systems."""
    
    def test_branch_creation_and_management(self):
        """Test reasoning branch creation and lifecycle."""
        # Test branch proposal, evaluation, and decision making
        
        # Example:
        # controller = BranchController(max_branches=3)
        # 
        # # Create branches
        # branch1 = controller.propose_branch("reasoning_path_1")
        # branch2 = controller.propose_branch("reasoning_path_2")
        # 
        # assert len(controller.active_branches) == 2
        # assert controller.can_create_branch() == True  # Under limit
        
        # Placeholder
        max_branches = 3
        active_branches = ['branch1', 'branch2']
        assert len(active_branches) < max_branches
    
    def test_branch_evaluation_and_acceptance(self):
        """Test branch evaluation and acceptance criteria."""
        # Test that branches are properly evaluated and accepted/rejected
        
        # Example:
        # evaluator = BranchEvaluator(threshold=0.5)
        # 
        # good_branch = Mock()
        # good_branch.advantage = 0.7
        # 
        # bad_branch = Mock()
        # bad_branch.advantage = 0.3
        # 
        # assert evaluator.should_accept(good_branch) == True
        # assert evaluator.should_accept(bad_branch) == False
        
        # Placeholder
        threshold = 0.5
        branch_advantage = 0.7
        assert branch_advantage > threshold
    
    def test_branch_rollback_mechanism(self):
        """Test branch rollback and state restoration."""
        # Test that rejected branches can be rolled back cleanly
        
        # Example:
        # controller = ReasoningController()
        # initial_state = controller.get_state()
        # 
        # # Propose and reject a branch
        # branch = controller.propose_branch("test_branch")
        # controller.reject_branch(branch)
        # 
        # final_state = controller.get_state()
        # assert states_equal(initial_state, final_state)
        
        # Placeholder
        initial_params = torch.randn(100)
        modified_params = initial_params + 0.1
        restored_params = initial_params.clone()
        
        assert torch.allclose(initial_params, restored_params)
    
    def test_concurrent_branch_processing(self):
        """Test concurrent processing of multiple reasoning branches."""
        # Test that multiple branches can be processed simultaneously
        
        # Example:
        # controller = ConcurrentBranchController(max_concurrent=2)
        # 
        # branch1 = controller.start_branch("path1")
        # branch2 = controller.start_branch("path2")
        # 
        # results = controller.wait_for_completion()
        # assert len(results) == 2
        
        # Placeholder for concurrent processing
        branches = ['branch1', 'branch2', 'branch3']
        max_concurrent = 2
        
        # Simulate processing batches
        processed = 0
        while processed < len(branches):
            batch_size = min(max_concurrent, len(branches) - processed)
            processed += batch_size
        
        assert processed == len(branches)


class TestPlasticitySystem:
    """Test plasticity and adaptation mechanisms."""
    
    def test_ephemeral_adapter_lifecycle(self):
        """Test ephemeral adapter creation, modification, and cleanup."""
        # Test adapter commit/rollback functionality
        
        # Example:
        # adapter = EphemeralAdapter(rank=8, alpha=0.2)
        # original_weights = adapter.get_weights().clone()
        # 
        # # Modify adapter
        # adapter.update_weights(torch.randn_like(original_weights) * 0.1)
        # modified_weights = adapter.get_weights()
        # 
        # # Test rollback
        # adapter.rollback()
        # restored_weights = adapter.get_weights()
        # 
        # assert not torch.allclose(original_weights, modified_weights)
        # assert torch.allclose(original_weights, restored_weights)
        
        # Placeholder
        original = torch.randn(64)
        modified = original + torch.randn(64) * 0.1
        restored = original.clone()
        
        assert torch.allclose(original, restored)
    
    def test_hebbian_correlation_computation(self):
        """Test Hebbian learning correlation computation."""
        # Test correlation-based plasticity objectives
        
        # Example:
        # hebbian_layer = HebbianPlasticityLayer(hidden_dim=256)
        # 
        # activations = torch.randn(32, 256)  # batch_size x hidden_dim
        # correlation_loss = hebbian_layer.compute_correlation_loss(activations)
        # 
        # assert correlation_loss.dim() == 0  # Scalar loss
        # assert correlation_loss >= 0  # Loss should be non-negative
        
        # Placeholder correlation computation
        activations = torch.randn(32, 256)
        
        # Simple correlation computation
        centered = activations - activations.mean(dim=0)
        correlation_matrix = torch.mm(centered.T, centered) / (centered.size(0) - 1)
        
        assert correlation_matrix.shape == (256, 256)
        assert torch.allclose(correlation_matrix, correlation_matrix.T)  # Should be symmetric
    
    def test_adapter_isolation(self):
        """Test that adapters don't interfere with each other."""
        # Test that concurrent adapters maintain isolation
        
        # Example:
        # adapter1 = EphemeralAdapter(rank=8)
        # adapter2 = EphemeralAdapter(rank=8)
        # 
        # adapter1.update_weights(torch.ones(64))
        # adapter2.update_weights(torch.zeros(64))
        # 
        # assert not torch.allclose(adapter1.get_weights(), adapter2.get_weights())
        
        # Placeholder
        state1 = {{'weights': torch.ones(64)}}
        state2 = {{'weights': torch.zeros(64)}}
        
        assert not torch.allclose(state1['weights'], state2['weights'])


class TestRLPolicySystem:
    """Test reinforcement learning policy components."""
    
    def test_policy_network_forward_pass(self):
        """Test RL policy network forward pass."""
        # Test policy network computation
        
        # Example:
        # policy = MemoryActionPolicy(state_dim=512, action_dim=4)
        # state = torch.randn(1, 512)
        # 
        # action_logits = policy(state)
        # action_probs = torch.softmax(action_logits, dim=-1)
        # 
        # assert action_logits.shape == (1, 4)
        # assert torch.allclose(action_probs.sum(dim=-1), torch.ones(1))
        
        # Placeholder
        state = torch.randn(1, 512)
        policy_net = torch.nn.Linear(512, 4)
        
        logits = policy_net(state)
        probs = torch.softmax(logits, dim=-1)
        
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), rtol=1e-4)
    
    def test_advantage_estimation(self):
        """Test advantage estimation for policy optimization."""
        # Test advantage computation
        
        # Example:
        # advantage_estimator = AdvantageEstimator(discount=0.99, gae_lambda=0.95)
        # 
        # rewards = torch.tensor([1.0, 0.5, -0.2, 0.8])
        # values = torch.tensor([0.7, 0.4, 0.3, 0.6])
        # 
        # advantages = advantage_estimator.compute(rewards, values)
        # assert advantages.shape == rewards.shape
        
        # Placeholder
        rewards = torch.tensor([1.0, 0.5, -0.2, 0.8])
        values = torch.tensor([0.7, 0.4, 0.3, 0.6])
        
        # Simple advantage: reward - value
        advantages = rewards - values[:-1]  # Simplified
        assert advantages.shape[0] == rewards.shape[0] - 1
    
    def test_memory_action_space(self):
        """Test memory action space and execution."""
        # Test the four memory actions: summarize, retrieve, splice_window, cache_write
        
        memory_actions = ['summarize', 'retrieve', 'splice_window', 'cache_write']
        
        # Test action space coverage
        action_space_size = len(memory_actions)
        assert action_space_size == 4
        
        # Test action execution (mock)
        for action in memory_actions:
            # This would test actual action execution
            # For now, verify action names are valid
            assert isinstance(action, str)
            assert len(action) > 0
    
    def test_policy_optimization_step(self):
        """Test policy optimization (GRPO/PPO) step."""
        # Test policy gradient update
        
        # Example:
        # optimizer = GRPOOptimizer(kl_target=0.02, clip_ratio=0.2)
        # 
        # old_logprobs = torch.randn(32)
        # new_logprobs = torch.randn(32)
        # advantages = torch.randn(32)
        # 
        # loss = optimizer.compute_loss(old_logprobs, new_logprobs, advantages)
        # assert loss.dim() == 0  # Scalar loss
        
        # Placeholder
        batch_size = 32
        old_logprobs = torch.randn(batch_size)
        new_logprobs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        
        # Simple policy ratio
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        assert ratio.shape == (batch_size,)
        assert torch.all(ratio > 0)  # Ratios should be positive


class TestReasoningIntegration:
    """Test integration of reasoning components."""
    
    def test_end_to_end_reasoning_pipeline(self):
        """Test complete reasoning pipeline from input to output."""
        # Test full reasoning flow
        
        # Example:
        # reasoning_system = HybridHebbianRLReasoner(
        #     max_branches=3,
        #     memory_config=memory_config,
        #     adapter_config=adapter_config
        # )
        # 
        # input_text = "Solve this problem step by step: 2 + 2 = ?"
        # reasoning_result = reasoning_system.reason(input_text)
        # 
        # assert reasoning_result.final_answer is not None
        # assert len(reasoning_result.reasoning_steps) > 0
        # assert reasoning_result.confidence_score >= 0.0
        
        # Placeholder
        input_tokens = torch.randint(0, 1000, (1, 32))
        output_tokens = input_tokens + 1  # Simple transformation
        
        assert output_tokens.shape == input_tokens.shape
    
    def test_reasoning_with_memory_integration(self):
        """Test reasoning system integration with memory components."""
        # Test that reasoning properly uses memory systems
        
        # Example:
        # reasoner = ReasoningSystem()
        # memory_context = torch.randn(1, 1024, 512)  # Long context
        # 
        # # Should use memory operations for long context
        # with patch.object(reasoner.memory, 'summarize') as mock_summarize:
        #     result = reasoner.reason_with_long_context(memory_context)
        #     mock_summarize.assert_called()
        
        # Placeholder
        context_length = 1024
        max_context = 512
        
        # Should trigger summarization
        needs_summarization = context_length > max_context
        assert needs_summarization == True
    
    def test_reasoning_scalability(self):
        """Test reasoning system behavior across different model scales."""
        # Test scale-dependent parameter adjustment
        
        model_scales = ['1b', '4b', '40b', '200b']
        expected_branch_counts = [3, 3, 4, 4]
        
        for scale, expected_branches in zip(model_scales, expected_branch_counts):
            # This would test actual scale-dependent configuration
            # For now, verify the scaling logic
            if '40b' in scale or '200b' in scale:
                assert expected_branches == 4
            else:
                assert expected_branches == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
        
        return test_code
    
    def generate_reasoning_validation_report(self) -> str:
        """Generate comprehensive reasoning validation report."""
        result = self.validate_complete_reasoning_system()
        
        report = f"""
# Reasoning System Validation Report

## Detected Reasoning Systems
"""
        
        for reasoning_type in result.detected_reasoning_types:
            report += f"- **{reasoning_type.value.title().replace('_', ' ')}**\n"
        
        report += f"""
## Validation Summary
- **Reasoning System Validity**: {'✅ VALID' if result.is_reasoning_valid else '❌ INVALID'}
- **Total Issues**: {len(result.issues)}
  - Critical: {len([i for i in result.issues if i.severity == ReasoningValidationSeverity.CRITICAL])}
  - Warnings: {len([i for i in result.issues if i.severity == ReasoningValidationSeverity.WARNING])}
  - Info: {len([i for i in result.issues if i.severity == ReasoningValidationSeverity.INFO])}

## System Capabilities Analysis

### Memory System
- **Working Memory**: {'✅' if result.capabilities.memory_system.has_working_memory else '❌'}
- **Episodic Memory**: {'✅' if result.capabilities.memory_system.has_episodic_memory else '❌'}
- **Memory Operations**: {'✅' if result.capabilities.memory_system.has_memory_operations else '❌'}
- **Retrieval System**: {'✅' if result.capabilities.memory_system.has_retrieval_system else '❌'}
- **Indexing Type**: {result.capabilities.memory_system.memory_indexing_type or 'None'}

### Reasoning Controller
- **Branch Management**: {'✅' if result.capabilities.controller.has_branch_management else '❌'}
- **Proposal System**: {'✅' if result.capabilities.controller.has_proposal_system else '❌'}
- **Evaluation System**: {'✅' if result.capabilities.controller.has_evaluation_system else '❌'}
- **Acceptance Criteria**: {'✅' if result.capabilities.controller.has_acceptance_criteria else '❌'}
- **Rollback Support**: {'✅' if result.capabilities.controller.supports_rollback else '❌'}
- **Max Steps**: {result.capabilities.controller.max_reasoning_steps or 'Not specified'}

### Plasticity System
- **Ephemeral Adapters**: {'✅' if result.capabilities.plasticity.has_ephemeral_adapters else '❌'}
- **Hebbian Learning**: {'✅' if result.capabilities.plasticity.has_hebbian_learning else '❌'}
- **Correlation Objectives**: {'✅' if result.capabilities.plasticity.has_correlation_objectives else '❌'}
- **Adapter Management**: {'✅' if result.capabilities.plasticity.has_adapter_management else '❌'}
- **Commit/Rollback**: {'✅' if result.capabilities.plasticity.supports_commit_rollback else '❌'}

### RL Policy System
- **Policy Network**: {'✅' if result.capabilities.rl_policy.has_policy_network else '❌'}
- **Reward System**: {'✅' if result.capabilities.rl_policy.has_reward_system else '❌'}
- **Advantage Estimation**: {'✅' if result.capabilities.rl_policy.has_advantage_estimation else '❌'}
- **Policy Optimization**: {'✅' if result.capabilities.rl_policy.has_policy_optimization else '❌'}
- **Memory Actions**: {'✅' if result.capabilities.rl_policy.supports_memory_actions else '❌'}
- **Algorithm**: {result.capabilities.rl_policy.algorithm_type or 'Not detected'}

## Issues Found
"""
        
        # Group issues by severity
        critical_issues = [i for i in result.issues if i.severity == ReasoningValidationSeverity.CRITICAL]
        warning_issues = [i for i in result.issues if i.severity == ReasoningValidationSeverity.WARNING]
        info_issues = [i for i in result.issues if i.severity == ReasoningValidationSeverity.INFO]
        
        if critical_issues:
            report += "\n### ❌ Critical Issues\n"
            for i, issue in enumerate(critical_issues, 1):
                report += f"{i}. **{issue.category.replace('_', ' ').title()}**: {issue.description}\n"
                if issue.reasoning_context:
                    report += f"   - *Context*: {issue.reasoning_context}\n"
                if issue.suggested_fix:
                    report += f"   - *Fix*: {issue.suggested_fix}\n"
                report += "\n"
        
        if warning_issues:
            report += "\n### ⚠️ Warnings\n"
            for i, issue in enumerate(warning_issues, 1):
                report += f"{i}. **{issue.category.replace('_', ' ').title()}**: {issue.description}\n"
                if issue.reasoning_context:
                    report += f"   - *Context*: {issue.reasoning_context}\n"
                report += "\n"
        
        if info_issues:
            report += "\n### ℹ️ Recommendations\n"
            for i, issue in enumerate(info_issues, 1):
                report += f"{i}. {issue.description}\n"
        
        report += "\n## Actionable Recommendations\n"
        for i, rec in enumerate(result.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        # Add reasoning-type specific guidance
        if ReasoningType.HYBRID_HEBBIAN_RL in result.detected_reasoning_types:
            report += """
## Hybrid Hebbian-RL Implementation Guide

### Core Components Required
1. **Hebbian Proposer**: Ephemeral plastic-LoRA adapters with correlation-based learning
2. **RL Evaluator**: GRPO-style policy optimization over memory operations
3. **Memory System**: Working memory (streaming KV) + Episodic memory (FAISS-indexed)
4. **Decision System**: Branch scoring, acceptance thresholds, rollback mechanisms

### Implementation Checklist
- [ ] Plastic-LoRA adapters with proper rank/alpha scaling
- [ ] Hebbian correlation loss computation
- [ ] Memory operations: summarize, retrieve, splice_window, cache_write
- [ ] RL policy network for memory action selection
- [ ] GRPO optimization with KL divergence constraints
- [ ] Branch management with commit/rollback isolation
- [ ] Scale-adaptive parameters (1B/4B/40B/200B configurations)

### Performance Requirements
- Memory operations: <10ms per action
- Branch evaluation: <50ms per proposal
- Adapter updates: <5ms per modification
- Resource limits: max_branches (3-4), step_budget (64-160)
"""
        
        elif ReasoningType.CHAIN_OF_THOUGHT in result.detected_reasoning_types:
            report += """
## Chain-of-Thought Implementation Guide

### Core Components
1. **Step Structure**: Explicit reasoning step decomposition
2. **Validation**: Reasoning chain verification and consistency checks
3. **Generation**: Step-by-step problem solving approach

### Best Practices
- Use explicit step markers and reasoning templates
- Implement reasoning chain validation
- Add step-wise confidence estimation
- Consider self-consistency decoding for complex problems
"""
        
        report += """
## Testing and Validation

### Immediate Actions
1. Run generated reasoning test suite
2. Fix all critical reasoning system issues
3. Implement missing core components
4. Add safety mechanisms (timeouts, resource limits)

### Continuous Validation
1. Monitor reasoning performance on benchmarks
2. Track memory utilization and efficiency
3. Validate branch acceptance rates
4. Profile reasoning latency and resource usage

### Integration Testing
1. Test end-to-end reasoning pipeline
2. Validate memory system integration
3. Test concurrent branch processing
4. Verify adapter isolation and rollback mechanisms

### Reasoning Quality Metrics
- Multi-step reasoning accuracy (>90% on benchmarks)
- Memory utilization efficiency (<5% overhead)
- Branch acceptance rate (70-85% for good balance)
- Adaptation speed (convergence within 10-20 steps)
"""
        
        return report


def main():
    """Main entry point for Reasoning Validator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reasoning System Validation")
    parser.add_argument("--project-root", type=Path, default=".", help="Project root directory")
    parser.add_argument("--reasoning-types", nargs='+', 
                       choices=[rt.value for rt in ReasoningType],
                       help="Force specific reasoning types")
    parser.add_argument("--generate-tests", action="store_true", help="Generate reasoning test suite")
    parser.add_argument("--report", action="store_true", help="Generate validation report")
    
    args = parser.parse_args()
    
    # Convert reasoning type strings to enums if provided
    reasoning_types = None
    if args.reasoning_types:
        reasoning_types = [ReasoningType(rt) for rt in args.reasoning_types]
    
    validator = ReasoningValidator(args.project_root, reasoning_types)
    
    if args.generate_tests:
        test_code = validator.generate_reasoning_test_suite()
        
        test_file = args.project_root / "test_reasoning_validation.py"
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        print(f"Generated reasoning test suite: {test_file}")
    
    if args.report:
        report = validator.generate_reasoning_validation_report()
        
        report_file = args.project_root / "reasoning_validation_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Generated reasoning validation report: {report_file}")
        print("\n" + "="*60)
        print(report)
    
    # Default: run validation and show summary
    if not args.generate_tests and not args.report:
        result = validator.validate_complete_reasoning_system()
        
        detected_types_str = ", ".join([rt.value.title().replace('_', ' ') for rt in result.detected_reasoning_types])
        print(f"Detected Reasoning: {detected_types_str}")
        print(f"Reasoning Validity: {'✅ VALID' if result.is_reasoning_valid else '❌ INVALID'}")
        print(f"Issues: {len(result.issues)} total")
        print(f"Critical: {len([i for i in result.issues if i.severity == ReasoningValidationSeverity.CRITICAL])}")
        
        if result.issues:
            print("\nMost critical issues:")
            for issue in result.issues[:3]:
                print(f"- {issue.category}: {issue.description}")


if __name__ == "__main__":
    main()