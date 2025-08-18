"""
CDRmix Complete Integration Test Suite

Tests the full CDRmix model integrating:
- RWKV-X architecture scheduling
- MoE routing with expert scaling  
- Streaming computation
- Text generation capabilities

Validates the complete system works across different model scales.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any, List

from src.cdrmix.cdrmix_model import (
    CDRmixConfig, 
    CDRmixModel,
    create_cdrmix_1b,
    create_cdrmix_4b
)


class TestCDRmixConfig:
    """Test CDRmix configuration system."""
    
    def test_should_create_1b_config(self):
        """Test 1B model configuration."""
        config = CDRmixConfig(model_scale='1b')
        
        # Verify base parameters
        assert config.model_scale == '1b'
        assert config.d_model == 2048
        assert config.n_layers == 24
        assert config.n_heads == 16
        
        # Verify MoE parameters
        assert config.num_experts == 8
        assert config.top_k == 2
        assert config.expert_d_ff > config.d_model
        
        # Verify RWKV-X architecture distribution
        assert config.transformer_ratio == 0.25
        assert config.total_transformer_blocks == 6  # 25% of 24
        assert config.rwkv_blocks == 18  # 75% of 24
        assert config.top_of_x_blocks == 3  # 50% of 6
        assert config.interleave_blocks == 3  # 50% of 6
    
    def test_should_create_4b_config(self):
        """Test 4B model configuration."""
        config = CDRmixConfig(model_scale='4b')
        
        assert config.model_scale == '4b'
        assert config.d_model == 3072
        assert config.n_layers == 36
        assert config.n_heads == 24
        
        # Verify architecture scaling
        assert config.total_transformer_blocks == 9  # 25% of 36
        assert config.rwkv_blocks == 27  # 75% of 36
        assert config.expert_d_ff > config.d_model
    
    def test_should_handle_custom_parameters(self):
        """Test custom parameter overrides."""
        config = CDRmixConfig(
            model_scale='1b',
            transformer_ratio=0.30,
            top_of_x_ratio=0.75,
            num_experts=16,
            top_k=4
        )
        
        assert config.transformer_ratio == 0.30
        assert config.top_of_x_ratio == 0.75
        assert config.num_experts == 16
        assert config.top_k == 4
        
        # Verify computed values
        assert config.total_transformer_blocks == 7  # 30% of 24
        assert config.top_of_x_blocks == 5  # 75% of 7
        assert config.interleave_blocks == 2  # 25% of 7


class TestCDRmixModel:
    """Test complete CDRmix model functionality."""
    
    @pytest.fixture
    def model_1b(self):
        """Create 1B CDRmix model for testing."""
        return create_cdrmix_1b(vocab_size=1000)  # Smaller vocab for testing
    
    @pytest.fixture  
    def model_4b(self):
        """Create 4B CDRmix model for testing."""
        return create_cdrmix_4b(vocab_size=1000)  # Smaller vocab for testing
    
    def test_model_should_have_correct_architecture(self, model_1b):
        """Test model architecture matches configuration."""
        config = model_1b.config
        
        # Check layer count
        assert len(model_1b.layers) == config.n_layers
        
        # Check layer types
        rwkv_count = sum(1 for layer in model_1b.layers 
                        if hasattr(layer, 'time_mixing'))
        transformer_count = sum(1 for layer in model_1b.layers 
                              if hasattr(layer, 'attention'))
        
        assert rwkv_count == config.rwkv_blocks
        assert transformer_count == config.total_transformer_blocks
    
    def test_model_forward_pass(self, model_1b):
        """Test model forward computation."""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Test forward pass without aux losses
        logits = model_1b(input_ids, return_aux_losses=False, return_states=False)
        
        assert logits.shape == (batch_size, seq_len, 1000)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_model_forward_with_states_and_losses(self, model_1b):
        """Test model forward with RWKV states and MoE losses."""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Test forward with states and aux losses
        logits, states, aux_losses = model_1b(
            input_ids, return_aux_losses=True, return_states=True
        )
        
        assert logits.shape == (batch_size, seq_len, 1000)
        assert states is not None
        assert len(states) == len(model_1b.layers)
        assert aux_losses is not None
        assert 'load_balance' in aux_losses
        assert 'z_loss' in aux_losses
        assert 'overflow' in aux_losses
    
    def test_model_streaming_generation(self, model_1b):
        """Test streaming text generation."""
        # Create input prompt
        input_ids = torch.randint(0, 1000, (1, 8))
        
        # Generate text
        generated = model_1b.generate(
            input_ids, 
            max_length=20,
            temperature=1.0,
            do_sample=True
        )
        
        assert generated.shape[0] == 1
        assert generated.shape[1] <= 20
        assert generated.shape[1] > input_ids.shape[1]
    
    def test_model_deterministic_generation(self, model_1b):
        """Test deterministic generation with same seed."""
        input_ids = torch.randint(0, 1000, (1, 8))
        
        # Generate with same seed twice
        torch.manual_seed(42)
        gen1 = model_1b.generate(input_ids, max_length=15, do_sample=False)
        
        torch.manual_seed(42) 
        gen2 = model_1b.generate(input_ids, max_length=15, do_sample=False)
        
        assert torch.equal(gen1, gen2)
    
    def test_model_scaling_across_sizes(self):
        """Test model creation across different scales."""
        scales = ['1b', '4b']
        models = []
        
        for scale in scales:
            if scale == '1b':
                model = create_cdrmix_1b(vocab_size=1000)
            else:
                model = create_cdrmix_4b(vocab_size=1000)
            
            models.append(model)
            
            # Test forward pass
            input_ids = torch.randint(0, 1000, (1, 8))
            logits = model(input_ids, return_aux_losses=False, return_states=False)
            
            assert logits.shape == (1, 8, 1000)
            assert not torch.isnan(logits).any()
        
        # Verify larger model has more parameters
        params_1b = sum(p.numel() for p in models[0].parameters())
        params_4b = sum(p.numel() for p in models[1].parameters())
        assert params_4b > params_1b
    
    def test_model_statistics(self, model_1b):
        """Test model statistics generation."""
        input_ids = torch.randint(0, 1000, (2, 16))
        
        stats = model_1b.get_model_statistics(input_ids)
        
        # Check required statistics
        required_keys = [
            'model_scale', 'total_layers', 'rwkv_layers', 
            'transformer_layers', 'num_experts', 'expert_top_k',
            'total_parameters', 'trainable_parameters',
            'moe_layer_statistics'
        ]
        
        for key in required_keys:
            assert key in stats, f"Missing statistic: {key}"
        
        # Verify values match config
        assert stats['model_scale'] == '1b'
        assert stats['total_layers'] == 24
        assert stats['num_experts'] == 8
        assert stats['expert_top_k'] == 2
        assert stats['total_parameters'] > 0
        assert stats['trainable_parameters'] == stats['total_parameters']
        
        # Check MoE statistics
        assert len(stats['moe_layer_statistics']) > 0
        for moe_stat in stats['moe_layer_statistics']:
            assert 'layer_index' in moe_stat
            assert 'layer_type' in moe_stat
            assert moe_stat['layer_type'] in ['rwkv', 'transformer']


class TestCDRmixStreamingComputation:
    """Test streaming computation capabilities."""
    
    @pytest.fixture
    def model(self):
        """Create model for streaming tests."""
        return create_cdrmix_1b(vocab_size=1000)
    
    def test_streaming_vs_full_sequence_accuracy(self, model):
        """Test streaming computation matches full sequence."""
        input_ids = torch.randint(0, 1000, (1, 32))
        
        # Full sequence computation
        full_logits, full_states = model(
            input_ids, return_aux_losses=False, return_states=True
        )
        
        # Streaming computation (two chunks)
        chunk1 = input_ids[:, :16]
        chunk2 = input_ids[:, 16:]
        
        # First chunk
        logits1, states1 = model(
            chunk1, return_aux_losses=False, return_states=True
        )
        
        # Second chunk with states
        logits2, states2 = model(
            chunk2, rwkv_states=states1, return_aux_losses=False, return_states=True
        )
        
        # Combine streaming results
        streaming_logits = torch.cat([logits1, logits2], dim=1)
        
        # Compare accuracy
        mse_error = torch.mean((streaming_logits - full_logits) ** 2)
        max_error = torch.max(torch.abs(streaming_logits - full_logits))
        
        # Should be reasonably accurate
        assert mse_error < 1e-2, f"MSE error too high: {mse_error}"
        assert max_error < 1.0, f"Max error too high: {max_error}"
    
    def test_state_updates_during_streaming(self, model):
        """Test RWKV states are properly updated during streaming."""
        input_ids = torch.randint(0, 1000, (1, 16))
        
        # First forward pass
        _, states1 = model(input_ids, return_aux_losses=False, return_states=True)
        
        # Second forward pass with same input
        _, states2 = model(
            input_ids, rwkv_states=states1, return_aux_losses=False, return_states=True
        )
        
        # States should be updated (different)
        rwkv_state_indices = [i for i, state in enumerate(states1) if state is not None]
        
        for i in rwkv_state_indices:
            assert not torch.equal(states1[i], states2[i]), f"RWKV state {i} was not updated"


class TestCDRmixRWKVXIntegration:
    """Test integration with RWKV-X architecture patterns."""
    
    def test_layer_schedule_follows_rwkv_x_pattern(self):
        """Test layer scheduling follows RWKV-X patterns."""
        config = CDRmixConfig(model_scale='1b', transformer_ratio=0.25)
        model = CDRmixModel(config)
        
        # Get actual layer types
        layer_types = []
        for layer in model.layers:
            if hasattr(layer, 'time_mixing'):
                layer_types.append('rwkv')
            else:
                layer_types.append('transformer')
        
        # Check that transformer blocks are at top (top-of-x pattern)
        top_blocks = layer_types[:config.top_of_x_blocks]
        assert all(lt == 'transformer' for lt in top_blocks), "Top-of-X pattern not followed"
        
        # Count total blocks
        total_transformer = layer_types.count('transformer')
        total_rwkv = layer_types.count('rwkv')
        
        assert total_transformer == config.total_transformer_blocks
        assert total_rwkv == config.rwkv_blocks
    
    def test_custom_rwkv_x_ratios(self):
        """Test custom RWKV-X ratios work correctly."""
        # Test different transformer ratios
        ratios = [0.1, 0.25, 0.5, 0.75]
        
        for ratio in ratios:
            config = CDRmixConfig(model_scale='1b', transformer_ratio=ratio)
            model = CDRmixModel(config)
            
            # Count actual layer types
            transformer_count = sum(1 for layer in model.layers 
                                  if hasattr(layer, 'attention'))
            rwkv_count = sum(1 for layer in model.layers 
                            if hasattr(layer, 'time_mixing'))
            
            expected_transformer = int(24 * ratio)  # 24 layers in 1B model
            expected_rwkv = 24 - expected_transformer
            
            assert transformer_count == expected_transformer
            assert rwkv_count == expected_rwkv
            
            # Test forward pass works
            input_ids = torch.randint(0, 1000, (1, 8))
            logits = model(input_ids, return_aux_losses=False, return_states=False)
            assert not torch.isnan(logits).any()


class TestCDRmixFactoryFunctions:
    """Test factory functions for different model scales."""
    
    def test_factory_functions_create_correct_models(self):
        """Test factory functions create models with correct configurations."""
        factories = [
            (create_cdrmix_1b, '1b'),
            (create_cdrmix_4b, '4b')
        ]
        
        for factory_func, expected_scale in factories:
            model = factory_func(vocab_size=1000)
            
            assert model.config.model_scale == expected_scale
            assert model.config.vocab_size == 1000
            
            # Test forward pass
            input_ids = torch.randint(0, 1000, (1, 8))
            logits = model(input_ids, return_aux_losses=False, return_states=False)
            
            assert logits.shape == (1, 8, 1000)
            assert not torch.isnan(logits).any()
    
    def test_factory_functions_accept_custom_parameters(self):
        """Test factory functions accept custom parameters."""
        model = create_cdrmix_1b(
            vocab_size=2000,
            num_experts=16,
            top_k=4,
            transformer_ratio=0.3
        )
        
        assert model.config.vocab_size == 2000
        assert model.config.num_experts == 16
        assert model.config.top_k == 4
        assert model.config.transformer_ratio == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])