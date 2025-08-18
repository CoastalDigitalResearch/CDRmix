"""
RWKV Block Implementation Tests
Following TDD principles - these tests define the required behavior.
"""

import torch
import pytest
import math
from typing import Optional, Tuple


def test_rwkv_block_should_have_required_components():
    """Test that RWKV block has the required mathematical components."""
    from src.cdrmix.rwkv_block import RWKVBlock
    
    # FAIL: This should fail initially since RWKVBlock is just a placeholder
    d_model = 512
    block = RWKVBlock(d_model)
    
    # Required mathematical components from specifications
    assert hasattr(block, 'time_mixing'), "RWKV block must have TimeMix component"
    assert hasattr(block, 'channel_mixing'), "RWKV block must have ChannelMix component"
    
    # Required projection matrices for RWKV computation
    assert hasattr(block.time_mixing, 'receptance'), "TimeMix must have receptance projection"
    assert hasattr(block.time_mixing, 'weight'), "TimeMix must have weight projection" 
    assert hasattr(block.time_mixing, 'key'), "TimeMix must have key projection"
    assert hasattr(block.time_mixing, 'value'), "TimeMix must have value projection"


def test_rwkv_block_should_implement_forward_pass():
    """Test that RWKV block implements forward computation."""
    from src.cdrmix.rwkv_block import RWKVBlock
    
    d_model = 512
    seq_len = 16
    batch_size = 2
    
    block = RWKVBlock(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Should process input tensor
    output, state = block(x)
    
    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
    assert state is not None, "Block should return state for streaming"
    assert not torch.isnan(output).any(), "Output should not contain NaN"
    assert not torch.isinf(output).any(), "Output should not contain Inf"


def test_rwkv_time_mixing_should_implement_linear_complexity():
    """Test that TimeMix achieves O(n*d) complexity, not O(n²*d)."""
    from src.cdrmix.rwkv_block import RWKVBlock
    
    d_model = 256
    block = RWKVBlock(d_model)
    
    # Test different sequence lengths - complexity should scale linearly
    seq_lengths = [64, 128, 256]
    times = []
    
    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, d_model)
        
        # Measure computational cost (simplified)
        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start and end:
                start.record()
                _ = block(x)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                # CPU timing fallback
                import time
                start_time = time.time()
                _ = block(x)
                times.append((time.time() - start_time) * 1000)
    
    # Verify roughly linear scaling (allowing for some variance)
    if len(times) >= 2:
        ratio_1_2 = times[1] / times[0] if times[0] > 0 else float('inf')
        ratio_2_3 = times[2] / times[1] if len(times) > 2 and times[1] > 0 else ratio_1_2
        
        # Should be roughly 2x for 2x sequence length (linear scaling)
        # Allow 50% tolerance for measurement variance
        assert 1.5 < ratio_1_2 < 3.0, f"Scaling ratio {ratio_1_2} suggests non-linear complexity"


def test_rwkv_should_support_streaming_computation():
    """Test that RWKV supports incremental token processing."""
    from src.cdrmix.rwkv_block import RWKVBlock
    
    d_model = 512
    seq_len = 32
    
    block = RWKVBlock(d_model)
    x = torch.randn(1, seq_len, d_model)
    
    # Full sequence processing
    full_output, _ = block(x)
    
    # Incremental processing - should maintain state
    state = None
    incremental_outputs = []
    
    for i in range(seq_len):
        token = x[:, i:i+1, :]  # Single token
        output, state = block.forward_incremental(token, state)
        incremental_outputs.append(output)
    
    incremental_output = torch.cat(incremental_outputs, dim=1)
    
    # Results should be approximately equal (allowing for numerical precision)
    # RWKV streaming may have small numerical differences due to floating point accumulation
    assert torch.allclose(full_output, incremental_output, rtol=1e-3, atol=1e-4), \
        "Incremental processing should match full sequence processing"


def test_rwkv_mathematical_formulation():
    """Test the core RWKV mathematical computation: WKV = Σ(exp(W + K) · V) / Σ(exp(W + K))."""
    from src.cdrmix.rwkv_block import RWKVBlock
    
    d_model = 128  # Smaller for easier testing
    seq_len = 8
    batch_size = 1
    
    block = RWKVBlock(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Access TimeMix component to verify mathematical computation
    time_mix = block.time_mixing
    
    with torch.no_grad():
        # Extract RWKV components
        R = torch.sigmoid(time_mix.receptance(x))  # Receptance
        W = torch.exp(-torch.exp(time_mix.weight(x)))  # Weight (time decay)
        K = time_mix.key(x)  # Key
        V = time_mix.value(x)  # Value
        
        # Verify component shapes
        assert R.shape == x.shape, f"Receptance shape {R.shape} != input shape {x.shape}"
        assert W.shape == x.shape, f"Weight shape {W.shape} != input shape {x.shape}"
        assert K.shape == x.shape, f"Key shape {K.shape} != input shape {x.shape}" 
        assert V.shape == x.shape, f"Value shape {V.shape} != input shape {x.shape}"
        
        # Verify mathematical properties
        assert torch.all(R >= 0) and torch.all(R <= 1), "Receptance should be in [0,1] (sigmoid output)"
        assert torch.all(W > 0) and torch.all(W <= 1), "Weight should be in (0,1] (exponential decay)"
        
        # Verify the WKV computation can be performed
        # WKV = Σ(exp(W + K) · V) / Σ(exp(W + K))
        exp_wk = torch.exp(W + K)
        assert not torch.isnan(exp_wk).any(), "exp(W + K) should not contain NaN"
        assert not torch.isinf(exp_wk).any(), "exp(W + K) should not contain Inf"


def test_rwkv_channel_mixing_component():
    """Test that ChannelMix implements position-wise FFN with gating."""
    from src.cdrmix.rwkv_block import RWKVBlock
    
    d_model = 512
    seq_len = 16
    batch_size = 2
    
    block = RWKVBlock(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test ChannelMix component
    channel_mix = block.channel_mixing
    
    # Should have FFN projections
    assert hasattr(channel_mix, 'key_proj'), "ChannelMix should have key projection"
    assert hasattr(channel_mix, 'value_proj'), "ChannelMix should have value projection"
    assert hasattr(channel_mix, 'receptance_proj'), "ChannelMix should have receptance projection"
    
    # Test forward pass
    output = channel_mix(x)
    assert output.shape == x.shape, f"ChannelMix output shape {output.shape} != input shape {x.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])