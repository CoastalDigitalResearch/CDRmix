"""
RWKV-X Hybrid Architecture Implementation

Implements the RWKV-X pattern combining RWKV blocks (75%) with transformer blocks (25%)
using two scheduling strategies:
- Top-of-X: 12.5% transformer blocks stacked at the end
- Interleave-X: 12.5% transformer blocks distributed throughout

Mathematical complexity: O(0.25·L·n²·d + 0.75·L·n·d)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from enum import Enum
import math

from .rwkv_block import RWKVBlock
from .transformer_block import TransformerBlock


class BlockType(Enum):
    """Block types in RWKV-X architecture."""
    RWKV = "rwkv"
    TRANSFORMER = "transformer"


class ScheduleType(Enum):
    """RWKV-X scheduling strategies."""
    TOP_OF_X = "top_of_x"          # Transformer blocks at the end
    INTERLEAVE_X = "interleave_x"  # Transformer blocks distributed throughout
    HYBRID = "hybrid"              # Mix of both strategies


class RWKVXScheduler:
    """
    RWKV-X scheduler implementing the Lean specifications.
    
    Generates deterministic block sequences with exactly 25% transformer ratio
    split between top-of-x and interleave-x strategies.
    """
    
    def __init__(self, tx_ratio: float = 0.25):
        self.tx_ratio = tx_ratio
        assert 0.0 <= tx_ratio <= 1.0, f"tx_ratio {tx_ratio} must be in [0, 1]"
    
    def schedule_top_of_x(self, num_layers: int) -> List[BlockType]:
        """
        Generate top-of-x schedule: transformer blocks stacked at the end.
        
        From Lean spec: scheduleTopOfX places ⌊txRatio·L⌋ Tx blocks at the end.
        
        Args:
            num_layers: Total number of layers L
            
        Returns:
            List of block types with Tx blocks at the end
        """
        num_tx = int(self.tx_ratio * num_layers)
        num_rwkv = num_layers - num_tx
        
        schedule = [BlockType.RWKV] * num_rwkv + [BlockType.TRANSFORMER] * num_tx
        
        assert len(schedule) == num_layers, f"Schedule length {len(schedule)} != {num_layers}"
        assert schedule.count(BlockType.TRANSFORMER) == num_tx, f"Tx count mismatch"
        
        return schedule
    
    def schedule_interleave_x(self, num_layers: int, k: int = 4) -> List[BlockType]:
        """
        Generate interleave-x schedule: transformer block every k layers.
        
        From Lean spec: scheduleInterleave places Tx at positions where (i+1) % k == 0.
        
        Args:
            num_layers: Total number of layers L
            k: Interleaving frequency (default 4 for 25% ratio)
            
        Returns:
            List of block types with Tx blocks every k positions
        """
        schedule = []
        for i in range(num_layers):
            if (i + 1) % k == 0:
                schedule.append(BlockType.TRANSFORMER)
            else:
                schedule.append(BlockType.RWKV)
        
        expected_tx = num_layers // k
        actual_tx = schedule.count(BlockType.TRANSFORMER)
        
        assert len(schedule) == num_layers, f"Schedule length {len(schedule)} != {num_layers}"
        assert actual_tx == expected_tx, f"Tx count {actual_tx} != expected {expected_tx}"
        
        return schedule
    
    def schedule_hybrid(self, num_layers: int, k: int = 8) -> List[BlockType]:
        """
        Generate hybrid schedule: 12.5% top-of-x + 12.5% interleave-x = 25% total.
        
        This implements your specific requirement of splitting 25% transformer blocks
        between both strategies for optimal modeling capacity.
        
        Args:
            num_layers: Total number of layers L
            k: Interleaving frequency for interleave portion
            
        Returns:
            List of block types combining both strategies
        """
        total_tx = int(self.tx_ratio * num_layers)
        
        # Split transformer blocks: half for interleave, half for top-of-x
        interleave_tx = total_tx // 2
        top_tx = total_tx - interleave_tx
        
        # Create base RWKV schedule
        schedule = [BlockType.RWKV] * num_layers
        
        # Add interleave transformer blocks (every k positions, limited count)
        tx_placed = 0
        for i in range(num_layers):
            if (i + 1) % k == 0 and tx_placed < interleave_tx:
                schedule[i] = BlockType.TRANSFORMER
                tx_placed += 1
        
        # Add top-of-x transformer blocks at the end
        for i in range(top_tx):
            # Replace RWKV blocks at the end with transformer blocks
            end_pos = num_layers - 1 - i
            if schedule[end_pos] == BlockType.RWKV:  # Don't overwrite interleave Tx
                schedule[end_pos] = BlockType.TRANSFORMER
        
        actual_tx = schedule.count(BlockType.TRANSFORMER)
        assert len(schedule) == num_layers, f"Schedule length mismatch"
        assert actual_tx <= total_tx, f"Too many Tx blocks: {actual_tx} > {total_tx}"
        
        return schedule
    
    def get_complexity_analysis(self, num_layers: int, seq_len: int, d_model: int, schedule: List[BlockType]) -> dict:
        """
        Analyze computational complexity of a schedule.
        
        Based on Lean spec complexity bounds:
        - RWKV: O(n*d) per layer  
        - Transformer: O(n²*d) per layer
        
        Args:
            num_layers: Number of layers
            seq_len: Sequence length n
            d_model: Model dimension d
            schedule: Block type schedule
            
        Returns:
            Dictionary with complexity analysis
        """
        num_rwkv = schedule.count(BlockType.RWKV)
        num_tx = schedule.count(BlockType.TRANSFORMER)
        
        # Complexity constants from Lean spec
        rwkv_cost_per_layer = 7 * d_model * d_model + seq_len * d_model  # 7*d² + n*d
        tx_cost_per_layer = 10 * d_model * d_model + seq_len * seq_len * d_model  # 10*d² + n²*d
        
        total_rwkv_cost = num_rwkv * rwkv_cost_per_layer
        total_tx_cost = num_tx * tx_cost_per_layer
        total_cost = total_rwkv_cost + total_tx_cost
        
        return {
            "num_layers": num_layers,
            "num_rwkv": num_rwkv,
            "num_tx": num_tx,
            "tx_ratio": num_tx / num_layers,
            "rwkv_cost": total_rwkv_cost,
            "tx_cost": total_tx_cost,
            "total_cost": total_cost,
            "linear_component": total_rwkv_cost / total_cost,
            "quadratic_component": total_tx_cost / total_cost,
            "schedule": [b.value for b in schedule]
        }


class RWKVXBlock(nn.Module):
    """
    Unified block that can operate as either RWKV or Transformer.
    
    This allows dynamic block type switching in the RWKV-X architecture
    while maintaining consistent interfaces.
    """
    
    def __init__(
        self, 
        d_model: int,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        block_type: BlockType = BlockType.RWKV
    ):
        super().__init__()
        self.d_model = d_model
        self.block_type = block_type
        
        # Initialize both block types
        self.rwkv_block = RWKVBlock(d_model, d_ff)
        self.transformer_block = TransformerBlock(d_model, n_heads, d_ff, dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass using the configured block type.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            state: Optional RWKV state (ignored for transformer)
            mask: Optional attention mask (ignored for RWKV)
            
        Returns:
            For RWKV: (output, new_state)
            For Transformer: output
        """
        if self.block_type == BlockType.RWKV:
            return self.rwkv_block(x, state)
        else:  # TRANSFORMER
            output = self.transformer_block(x, mask)
            return output, None  # Return None state for consistency
    
    def set_block_type(self, block_type: BlockType):
        """Change the active block type."""
        self.block_type = block_type


class RWKVXArchitecture(nn.Module):
    """
    Complete RWKV-X hybrid architecture.
    
    Implements the full RWKV-X pattern with 75% RWKV blocks and 25% transformer blocks
    arranged according to the specified scheduling strategy.
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        schedule_type: ScheduleType = ScheduleType.HYBRID,
        tx_ratio: float = 0.25
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.schedule_type = schedule_type
        
        # Generate block schedule
        scheduler = RWKVXScheduler(tx_ratio)
        if schedule_type == ScheduleType.TOP_OF_X:
            self.schedule = scheduler.schedule_top_of_x(num_layers)
        elif schedule_type == ScheduleType.INTERLEAVE_X:
            self.schedule = scheduler.schedule_interleave_x(num_layers)
        else:  # HYBRID
            self.schedule = scheduler.schedule_hybrid(num_layers)
        
        # Create blocks according to schedule
        self.blocks = nn.ModuleList()
        for block_type in self.schedule:
            if block_type == BlockType.RWKV:
                block = RWKVBlock(d_model, d_ff)
            else:  # TRANSFORMER
                block = TransformerBlock(d_model, n_heads, d_ff, dropout)
            self.blocks.append(block)
        
        # Store complexity analysis
        self.complexity_analysis = scheduler.get_complexity_analysis(
            num_layers, seq_len=2048, d_model=d_model, schedule=self.schedule
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_layer_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through RWKV-X architecture.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask for transformer blocks
            return_layer_outputs: Whether to return intermediate outputs
            
        Returns:
            output: Final output tensor
            layer_outputs: List of layer outputs (if requested)
        """
        layer_outputs = []
        states = {}  # Track RWKV states per layer
        
        for i, (block, block_type) in enumerate(zip(self.blocks, self.schedule)):
            if block_type == BlockType.RWKV:
                # RWKV block with state tracking
                state = states.get(i, None)
                x, new_state = block(x, state)
                states[i] = new_state
            else:
                # Transformer block
                x = block(x, mask)
            
            if return_layer_outputs:
                layer_outputs.append(x.clone())
        
        if return_layer_outputs:
            return x, layer_outputs
        return x
    
    def get_schedule_info(self) -> dict:
        """Get information about the block schedule."""
        num_rwkv = self.schedule.count(BlockType.RWKV)
        num_tx = self.schedule.count(BlockType.TRANSFORMER)
        
        return {
            "total_layers": self.num_layers,
            "num_rwkv": num_rwkv,
            "num_transformer": num_tx,
            "rwkv_ratio": num_rwkv / self.num_layers,
            "tx_ratio": num_tx / self.num_layers,
            "schedule_type": self.schedule_type.value,
            "schedule": [b.value for b in self.schedule],
            "complexity_analysis": self.complexity_analysis
        }
    
    def forward_with_layer_analysis(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass with detailed layer-by-layer analysis.
        
        Returns detailed information about processing at each layer.
        """
        batch_size, seq_len, d_model = x.shape
        layer_info = []
        states = {}
        
        for i, (block, block_type) in enumerate(zip(self.blocks, self.schedule)):
            # Record layer info
            layer_data = {
                "layer": i,
                "block_type": block_type.value,
                "input_norm": x.norm().item()
            }
            
            # Forward pass
            if block_type == BlockType.RWKV:
                state = states.get(i, None)
                x, new_state = block(x, state)
                states[i] = new_state
                layer_data["has_state"] = True
                layer_data["state_norm"] = new_state.norm().item() if new_state is not None else 0.0
            else:
                x = block(x, mask)
                layer_data["has_state"] = False
                layer_data["state_norm"] = 0.0
            
            layer_data["output_norm"] = x.norm().item()
            layer_info.append(layer_data)
        
        return {
            "final_output": x,
            "layer_analysis": layer_info,
            "schedule_info": self.get_schedule_info()
        }