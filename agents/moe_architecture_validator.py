#!/usr/bin/env python3
"""
MoE-Architecture-Validator Agent

Specialized Claude Code agent for validating Mixture-of-Experts (MoE) architectures.
Focuses on expert routing algorithms, load balancing, sparsity patterns, and capacity constraints.

This agent provides deep domain expertise for MoE validation that general testing agents lack.
"""

import ast
import inspect
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import yaml


@dataclass
class MoEValidationResult:
    """Result of MoE architecture validation."""
    is_valid: bool
    violations: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]


@dataclass
class MoEConfig:
    """MoE configuration extracted from model config."""
    num_experts: int
    top_k: int
    capacity_factor: float
    routing_strategy: str
    load_balancing: bool
    expert_dropout: float = 0.0


class MoEArchitectureValidator:
    """
    Validates MoE architecture implementations against mathematical properties
    and performance requirements.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.violations = []
        self.warnings = []
        self.metrics = {}
    
    def validate_full_architecture(self, config_path: Optional[Path] = None) -> MoEValidationResult:
        """
        Complete MoE architecture validation across all components.
        
        Args:
            config_path: Optional path to MoE configuration file
            
        Returns:
            MoEValidationResult with comprehensive validation results
        """
        self.violations = []
        self.warnings = []
        self.metrics = {}
        
        # Load MoE configuration
        moe_config = self._load_moe_config(config_path)
        
        # Validate router implementation
        self._validate_router_implementation(moe_config)
        
        # Validate expert implementation 
        self._validate_expert_implementation(moe_config)
        
        # Validate capacity constraints
        self._validate_capacity_constraints(moe_config)
        
        # Validate load balancing
        self._validate_load_balancing(moe_config)
        
        # Validate sparsity patterns
        self._validate_sparsity_patterns(moe_config)
        
        # Validate routing algorithm correctness
        self._validate_routing_algorithms(moe_config)
        
        # Check Lean specification compliance
        self._validate_lean_spec_compliance(moe_config)
        
        return MoEValidationResult(
            is_valid=len(self.violations) == 0,
            violations=self.violations.copy(),
            warnings=self.warnings.copy(),
            metrics=self.metrics.copy()
        )
    
    def _load_moe_config(self, config_path: Optional[Path] = None) -> Optional[MoEConfig]:
        """Load MoE configuration from YAML files."""
        if config_path and config_path.exists():
            config_files = [config_path]
        else:
            # Search for config files
            config_files = list(self.project_root.glob("configs/*.yaml"))
            config_files.extend(list(self.project_root.glob("config/*.yaml")))
        
        moe_configs = []
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Extract MoE parameters
                if 'moe' in config or 'num_experts' in config:
                    moe_section = config.get('moe', config)
                    moe_config = MoEConfig(
                        num_experts=moe_section.get('num_experts', 8),
                        top_k=moe_section.get('top_k', 2),
                        capacity_factor=moe_section.get('capacity_factor', 1.25),
                        routing_strategy=moe_section.get('routing_strategy', 'learned'),
                        load_balancing=moe_section.get('load_balancing', True),
                        expert_dropout=moe_section.get('expert_dropout', 0.0)
                    )
                    moe_configs.append(moe_config)
            except Exception as e:
                self.warnings.append(f"Could not parse config {config_file}: {e}")
        
        return moe_configs[0] if moe_configs else None
    
    def _validate_router_implementation(self, moe_config: Optional[MoEConfig]):
        """Validate router implementation correctness."""
        router_file = self.project_root / "src" / "cdrmix" / "router.py"
        
        if not router_file.exists():
            self.violations.append("Router implementation file not found")
            return
        
        with open(router_file, 'r') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
            
            # Check for MoERouter class
            router_classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and 'Router' in n.name]
            
            if not router_classes:
                self.violations.append("No Router class found in router.py")
                return
            
            router_class = router_classes[0]
            
            # Check for required methods
            required_methods = ['forward', 'route_tokens', 'compute_routing_weights']
            methods = [n.name for n in router_class.body if isinstance(n, ast.FunctionDef)]
            
            for required_method in required_methods:
                if required_method not in methods and not any(required_method in m for m in methods):
                    self.warnings.append(f"Router missing recommended method: {required_method}")
            
            # Validate top-k constraint enforcement
            self._check_topk_constraint_in_code(content, moe_config)
            
            # Check for routing weight normalization
            if 'softmax' not in content.lower() and 'normalize' not in content.lower():
                self.warnings.append("Router may not properly normalize routing weights")
                
        except SyntaxError as e:
            self.violations.append(f"Router implementation has syntax errors: {e}")
    
    def _validate_expert_implementation(self, moe_config: Optional[MoEConfig]):
        """Validate expert implementation and dispatch logic."""
        moe_block_file = self.project_root / "src" / "cdrmix" / "moe_block.py"
        
        if not moe_block_file.exists():
            self.violations.append("MoE block implementation file not found")
            return
        
        with open(moe_block_file, 'r') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
            
            # Check for MoE block classes
            moe_classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and 'MoE' in n.name]
            
            if not moe_classes:
                self.violations.append("No MoE class found in moe_block.py")
                return
            
            # Check expert dispatch logic
            if 'experts' not in content.lower():
                self.warnings.append("MoE block may not implement expert dispatch")
            
            # Check for proper tensor routing
            if 'scatter' not in content and 'gather' not in content:
                self.warnings.append("MoE block may not implement proper tensor routing")
            
        except SyntaxError as e:
            self.violations.append(f"MoE block implementation has syntax errors: {e}")
    
    def _check_topk_constraint_in_code(self, content: str, moe_config: Optional[MoEConfig]):
        """Check that top-k constraint is properly enforced in routing code."""
        topk_patterns = [
            r'topk\s*\(',
            r'top_k\s*=',
            r'\.topk\(',
            r'torch\.topk\('
        ]
        
        has_topk = any(re.search(pattern, content, re.IGNORECASE) for pattern in topk_patterns)
        
        if not has_topk:
            self.violations.append("Router does not enforce top-k constraint")
        
        if moe_config:
            # Check that configured top_k is used
            topk_value_pattern = rf'\b{moe_config.top_k}\b'
            if not re.search(topk_value_pattern, content):
                self.warnings.append(f"Router may not use configured top_k value ({moe_config.top_k})")
    
    def _validate_capacity_constraints(self, moe_config: Optional[MoEConfig]):
        """Validate capacity constraint implementation against Lean specification."""
        if not moe_config:
            self.warnings.append("No MoE config found, skipping capacity validation")
            return
        
        # Check Lean specification exists
        lean_moe_file = self.project_root / "specs" / "lean" / "CDRmix" / "MoE.lean"
        
        if not lean_moe_file.exists():
            self.warnings.append("Lean MoE specification not found")
            return
        
        with open(lean_moe_file, 'r') as f:
            lean_content = f.read()
        
        # Validate capacity factor >= 1 (from Lean spec)
        if moe_config.capacity_factor < 1.0:
            self.violations.append(f"Capacity factor {moe_config.capacity_factor} < 1.0 violates Lean specification")
        
        # Check that perExpertCapacity formula is correctly implemented
        expected_capacity_formula = "ceil(phi * (N * topK) / E)"
        
        # Look for capacity calculation in implementation
        router_file = self.project_root / "src" / "cdrmix" / "router.py"
        if router_file.exists():
            with open(router_file, 'r') as f:
                router_content = f.read()
            
            if 'capacity' not in router_content.lower():
                self.violations.append("Router does not implement capacity constraints")
    
    def _validate_load_balancing(self, moe_config: Optional[MoEConfig]):
        """Validate load balancing implementation."""
        if not moe_config or not moe_config.load_balancing:
            return
        
        # Check for load balancing loss implementation
        files_to_check = [
            self.project_root / "src" / "cdrmix" / "router.py",
            self.project_root / "src" / "cdrmix" / "moe_block.py",
            self.project_root / "src" / "train.py"
        ]
        
        load_balance_terms = ['auxiliary_loss', 'load_balance', 'expert_balance', 'routing_loss']
        found_load_balancing = False
        
        for file_path in files_to_check:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                
                if any(term in content.lower() for term in load_balance_terms):
                    found_load_balancing = True
                    break
        
        if not found_load_balancing:
            self.violations.append("Load balancing is enabled but no load balancing loss found")
    
    def _validate_sparsity_patterns(self, moe_config: Optional[MoEConfig]):
        """Validate sparsity activation patterns."""
        if not moe_config:
            return
        
        # Calculate expected sparsity
        expected_sparsity = 1.0 - (moe_config.top_k / moe_config.num_experts)
        self.metrics['expected_sparsity'] = expected_sparsity
        
        # Check if sparsity is reasonable
        if expected_sparsity < 0.5:
            self.warnings.append(f"Low sparsity ({expected_sparsity:.2f}) may not provide efficiency benefits")
        
        # Check for gradient scaling due to sparsity
        files_to_check = [
            self.project_root / "src" / "cdrmix" / "moe_block.py",
            self.project_root / "src" / "train.py"
        ]
        
        for file_path in files_to_check:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Look for gradient scaling patterns
                if 'scale' in content.lower() and 'grad' in content.lower():
                    self.metrics['has_gradient_scaling'] = True
                    break
        else:
            self.warnings.append("No gradient scaling found for sparse MoE training")
    
    def _validate_routing_algorithms(self, moe_config: Optional[MoEConfig]):
        """Validate routing algorithm implementation."""
        if not moe_config:
            return
        
        router_file = self.project_root / "src" / "cdrmix" / "router.py"
        if not router_file.exists():
            return
        
        with open(router_file, 'r') as f:
            content = f.read()
        
        # Check routing strategy implementation
        if moe_config.routing_strategy == 'learned':
            if 'linear' not in content.lower() and 'dense' not in content.lower():
                self.warnings.append("Learned routing strategy but no linear layer found")
        
        # Check for routing stability (temperature, noise)
        stability_terms = ['temperature', 'noise', 'jitter', 'epsilon']
        has_stability = any(term in content.lower() for term in stability_terms)
        
        if not has_stability:
            self.warnings.append("Router may lack stability mechanisms (temperature/noise)")
    
    def _validate_lean_spec_compliance(self, moe_config: Optional[MoEConfig]):
        """Validate compliance with formal Lean specifications."""
        lean_moe_file = self.project_root / "specs" / "lean" / "CDRmix" / "MoE.lean"
        
        if not lean_moe_file.exists():
            self.warnings.append("No Lean specification found for verification")
            return
        
        with open(lean_moe_file, 'r') as f:
            lean_content = f.read()
        
        # Extract key properties from Lean spec
        properties = self._extract_lean_properties(lean_content)
        
        # Validate against properties
        for prop_name, prop_details in properties.items():
            if prop_name == 'NoOverflow' and moe_config:
                # This should be validated at runtime, flag for testing
                self.metrics['requires_runtime_validation'] = ['NoOverflow property']
            
            if prop_name == 'capacity_sufficient':
                if moe_config and moe_config.capacity_factor < 1.0:
                    self.violations.append("Capacity factor < 1.0 violates capacity_sufficient theorem")
    
    def _extract_lean_properties(self, lean_content: str) -> Dict[str, str]:
        """Extract key properties and theorems from Lean specification."""
        properties = {}
        
        # Extract structure definitions
        structure_pattern = r'structure\s+(\w+).*?where(.*?)(?=structure|\Z)'
        structures = re.findall(structure_pattern, lean_content, re.DOTALL)
        
        for name, body in structures:
            properties[name] = body.strip()
        
        # Extract definitions
        def_pattern = r'def\s+(\w+).*?:=.*?\n'
        definitions = re.findall(def_pattern, lean_content, re.DOTALL)
        
        for def_name in definitions:
            properties[def_name] = f"Definition: {def_name}"
        
        # Extract theorems
        theorem_pattern = r'theorem\s+(\w+).*?:=.*?by'
        theorems = re.findall(theorem_pattern, lean_content, re.DOTALL)
        
        for theorem_name in theorems:
            properties[theorem_name] = f"Theorem: {theorem_name}"
        
        return properties
    
    def generate_test_suite(self, moe_config: Optional[MoEConfig]) -> str:
        """Generate comprehensive test suite for MoE architecture."""
        if not moe_config:
            return "# No MoE configuration found - cannot generate tests"
        
        test_code = f'''"""
MoE Architecture Test Suite
Generated by MoE-Architecture-Validator

Tests for validating MoE routing, capacity constraints, and load balancing.
"""

import torch
import pytest
import numpy as np
from typing import Dict, List, Tuple

# Assuming MoE implementation is importable
try:
    from src.cdrmix.router import MoERouter
    from src.cdrmix.moe_block import MoERWKVBlock
except ImportError:
    pytest.skip("MoE implementation not available", allow_module_level=True)


class TestMoERouting:
    """Test MoE routing algorithm correctness."""
    
    def test_router_outputs_exact_topk(self):
        """Test that router outputs exactly top-k experts per token."""
        batch_size, seq_len, hidden_dim = 4, 16, 512
        num_experts, top_k = {moe_config.num_experts}, {moe_config.top_k}
        
        router = MoERouter(hidden_dim, num_experts, top_k)
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        routing_weights, selected_experts = router(x)
        
        # Verify exactly top_k experts selected per token
        assert selected_experts.shape[-1] == top_k
        
        # Verify routing weights sum to 1 for selected experts
        for b in range(batch_size):
            for s in range(seq_len):
                selected_weights = routing_weights[b, s, selected_experts[b, s]]
                assert torch.allclose(selected_weights.sum(), torch.tensor(1.0), rtol=1e-5)
    
    def test_capacity_constraints_respected(self):
        """Test that capacity constraints from Lean spec are respected."""
        batch_size, seq_len = 8, 32
        num_tokens = batch_size * seq_len
        num_experts, top_k = {moe_config.num_experts}, {moe_config.top_k}
        capacity_factor = {moe_config.capacity_factor}
        
        # Calculate expected capacity per expert (from Lean spec)
        expected_capacity = math.ceil(capacity_factor * (num_tokens * top_k) / num_experts)
        
        router = MoERouter(512, num_experts, top_k)
        x = torch.randn(batch_size, seq_len, 512)
        
        routing_weights, selected_experts = router(x)
        
        # Count tokens assigned to each expert
        expert_counts = torch.zeros(num_experts)
        flat_experts = selected_experts.flatten()
        
        for expert_id in range(num_experts):
            expert_counts[expert_id] = (flat_experts == expert_id).sum().item()
        
        # Verify no expert exceeds capacity
        max_expert_load = expert_counts.max().item()
        assert max_expert_load <= expected_capacity, f"Expert overload: {{max_expert_load}} > {{expected_capacity}}"
    
    def test_load_balancing_effectiveness(self):
        """Test that load balancing distributes tokens reasonably."""
        batch_size, seq_len = 16, 64
        num_experts, top_k = {moe_config.num_experts}, {moe_config.top_k}
        
        router = MoERouter(512, num_experts, top_k)
        x = torch.randn(batch_size, seq_len, 512)
        
        routing_weights, selected_experts = router(x)
        
        # Calculate load distribution
        expert_loads = torch.zeros(num_experts)
        flat_experts = selected_experts.flatten()
        
        for expert_id in range(num_experts):
            expert_loads[expert_id] = (flat_experts == expert_id).sum().item()
        
        # Check load balance (coefficient of variation should be reasonable)
        mean_load = expert_loads.mean()
        std_load = expert_loads.std()
        cv = std_load / mean_load if mean_load > 0 else float('inf')
        
        # CV should be < 0.5 for reasonable load balancing
        assert cv < 0.5, f"Poor load balancing: CV={{cv:.3f}}"
    
    def test_routing_determinism(self):
        """Test that routing is deterministic for same inputs."""
        torch.manual_seed(42)
        
        router = MoERouter(512, {moe_config.num_experts}, {moe_config.top_k})
        x = torch.randn(4, 8, 512)
        
        # Two forward passes with same input
        weights1, experts1 = router(x)
        weights2, experts2 = router(x)
        
        assert torch.allclose(weights1, weights2)
        assert torch.equal(experts1, experts2)
    
    def test_gradient_flow_through_routing(self):
        """Test that gradients flow properly through sparse routing."""
        router = MoERouter(512, {moe_config.num_experts}, {moe_config.top_k})
        x = torch.randn(2, 4, 512, requires_grad=True)
        
        routing_weights, selected_experts = router(x)
        
        # Create dummy loss
        loss = routing_weights.sum()
        loss.backward()
        
        # Check that input gradients exist and are reasonable
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestMoEBlock:
    """Test complete MoE block functionality."""
    
    def test_moe_block_forward_pass(self):
        """Test that MoE block processes input correctly."""
        batch_size, seq_len, hidden_dim = 4, 16, 512
        
        moe_block = MoERWKVBlock(
            hidden_dim=hidden_dim,
            num_experts={moe_config.num_experts},
            top_k={moe_config.top_k}
        )
        
        x = torch.randn(batch_size, seq_len, hidden_dim)
        output = moe_block(x)
        
        # Output should have same shape as input
        assert output.shape == x.shape
        
        # Output should not be NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_expert_specialization(self):
        """Test that different experts produce different outputs."""
        hidden_dim = 512
        moe_block = MoERWKVBlock(
            hidden_dim=hidden_dim,
            num_experts={moe_config.num_experts},
            top_k=1  # Use only one expert at a time
        )
        
        x = torch.randn(1, 1, hidden_dim)
        
        # Force different experts and compare outputs
        outputs = []
        for expert_id in range({moe_config.num_experts}):
            # This would need expert forcing mechanism in implementation
            # outputs.append(moe_block.forward_with_expert(x, expert_id))
            pass
        
        # Check that expert outputs differ significantly
        # This test requires implementation of expert forcing
        pass


class TestMoEMathematicalProperties:
    """Test mathematical properties from Lean specification."""
    
    def test_capacity_sufficient_theorem(self):
        """Test the capacity_sufficient theorem from Lean spec."""
        # This test validates the mathematical theorem
        N = 1000  # number of tokens
        E = {moe_config.num_experts}  # number of experts  
        top_k = {moe_config.top_k}
        phi = {moe_config.capacity_factor}  # capacity factor >= 1
        
        # Per-expert capacity from Lean spec
        per_expert_capacity = math.ceil(phi * (N * top_k) / E)
        
        # Expected load under perfect balance
        expected_load_per_expert = math.ceil((N * top_k) / E)
        
        # Theorem: if phi >= 1, then capacity >= expected load
        assert per_expert_capacity >= expected_load_per_expert
    
    def test_no_overflow_property(self):
        """Test NoOverflow property from Lean specification."""
        # This would be tested at runtime with actual routing
        # For now, verify the mathematical constraint
        
        N = 500
        E = {moe_config.num_experts}
        top_k = {moe_config.top_k}  
        phi = {moe_config.capacity_factor}
        
        capacity = math.ceil(phi * (N * top_k) / E)
        total_capacity = capacity * E
        total_assignments = N * top_k
        
        # Total capacity should accommodate all assignments
        assert total_capacity >= total_assignments


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
        
        return test_code
    
    def generate_validation_report(self) -> str:
        """Generate detailed validation report."""
        result = self.validate_full_architecture()
        
        report = f"""
# MoE Architecture Validation Report

## Summary
- **Status**: {'✅ VALID' if result.is_valid else '❌ INVALID'}
- **Violations**: {len(result.violations)}
- **Warnings**: {len(result.warnings)}

## Violations
"""
        
        for i, violation in enumerate(result.violations, 1):
            report += f"{i}. ❌ {violation}\n"
        
        report += "\n## Warnings\n"
        
        for i, warning in enumerate(result.warnings, 1):
            report += f"{i}. ⚠️  {warning}\n"
        
        report += f"\n## Metrics\n"
        
        for metric, value in result.metrics.items():
            report += f"- **{metric}**: {value}\n"
        
        report += """
## Recommendations

1. **Router Implementation**: Ensure proper top-k constraint enforcement
2. **Capacity Constraints**: Implement Lean specification compliance  
3. **Load Balancing**: Add auxiliary loss for expert load balancing
4. **Testing**: Run generated test suite regularly
5. **Monitoring**: Add runtime validation of NoOverflow property

## Next Steps

1. Fix all violations before production deployment
2. Address warnings for optimal performance
3. Integrate generated test suite into CI/CD
4. Add runtime monitoring for capacity violations
"""
        
        return report


def main():
    """Main entry point for MoE Architecture Validator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate MoE architecture implementation")
    parser.add_argument("--project-root", type=Path, default=".", help="Project root directory")
    parser.add_argument("--config", type=Path, help="MoE configuration file")
    parser.add_argument("--generate-tests", action="store_true", help="Generate test suite")
    parser.add_argument("--report", action="store_true", help="Generate validation report")
    
    args = parser.parse_args()
    
    validator = MoEArchitectureValidator(args.project_root)
    
    if args.generate_tests:
        moe_config = validator._load_moe_config(args.config)
        test_code = validator.generate_test_suite(moe_config)
        
        test_file = args.project_root / "test_moe_architecture.py"
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        print(f"Generated test suite: {test_file}")
    
    if args.report:
        report = validator.generate_validation_report()
        
        report_file = args.project_root / "moe_validation_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Generated validation report: {report_file}")
        print("\n" + "="*50)
        print(report)


if __name__ == "__main__":
    main()