"""
MoE-RWKV Block Implementation for CDRmix

Combines RWKV blocks with MoE routing for expert specialization.
Supports both RWKV and Transformer experts in the RWKV-X architecture.

Expert sizes:
- 1B model: 8 x 125M parameter experts
- 4B model: 8 x 500M parameter experts  
- 40B model: 8 x 5B parameter experts
- 200B model: 8 x 25B parameter experts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List, Union
import math

from .router import MoERouter
from .rwkv_block import RWKVBlock, TimeMix, ChannelMix
from .transformer_block import TransformerBlock, TransformerFFN


class ExpertParameterCalculator:
    """
    Calculate expert parameters for different model scales.
    
    Ensures each model uses 8 experts with target parameter counts:
    - 1B total -> 125M per expert
    - 4B total -> 500M per expert
    - 40B total -> 5B per expert
    - 200B total -> 25B per expert
    """
    
    @staticmethod
    def calculate_expert_dimensions(
        total_model_params: int,
        num_experts: int = 8,
        d_model: int = 2048
    ) -> Dict[str, int]:
        """
        Calculate expert dimensions for target parameter count.
        
        Args:
            total_model_params: Target total parameters (1B, 4B, 40B, 200B)
            num_experts: Number of experts (8)
            d_model: Model dimension
            
        Returns:
            Dictionary with expert dimension specifications
        """
        target_params_per_expert = total_model_params // num_experts
        
        # For RWKV ChannelMix: d_ff = hidden dimension of expert FFN
        # Parameters in ChannelMix: key_proj (d_model * d_ff) + value_proj (d_ff * d_model) + receptance_proj (d_model * d_model)
        # Dominant term is 2 * d_model * d_ff
        # So: target_params ≈ 2 * d_model * d_ff
        # Therefore: d_ff ≈ target_params / (2 * d_model)
        
        expert_d_ff = max(target_params_per_expert // (2 * d_model), d_model)  # At least d_model
        
        # Calculate actual parameters
        channelmix_params = (
            d_model * expert_d_ff +  # key_proj
            expert_d_ff * d_model +  # value_proj  
            d_model * d_model        # receptance_proj
        )
        
        return {
            'expert_d_ff': expert_d_ff,
            'actual_params_per_expert': channelmix_params,
            'total_expert_params': channelmix_params * num_experts,
            'target_params_per_expert': target_params_per_expert
        }
    
    @staticmethod
    def get_model_scale_config(model_scale: str) -> Dict[str, Any]:
        """
        Get configuration for different model scales.
        
        Args:
            model_scale: '1b', '4b', '40b', or '200b'
            
        Returns:
            Configuration dictionary
        """
        scale_configs = {
            '1b': {
                'total_params': 1_000_000_000,  # 1B
                'd_model': 2048,
                'n_layers': 24,
                'n_heads': 16
            },
            '4b': {
                'total_params': 4_000_000_000,  # 4B
                'd_model': 3072,
                'n_layers': 36,
                'n_heads': 24
            },
            '40b': {
                'total_params': 40_000_000_000,  # 40B
                'd_model': 6144,
                'n_layers': 64,
                'n_heads': 48
            },
            '200b': {
                'total_params': 200_000_000_000,  # 200B
                'd_model': 12288,
                'n_layers': 96,
                'n_heads': 96
            }
        }
        
        if model_scale not in scale_configs:
            raise ValueError(f"Unknown model scale: {model_scale}. Options: {list(scale_configs.keys())}")
        
        config = scale_configs[model_scale]
        
        # Calculate expert dimensions
        expert_config = ExpertParameterCalculator.calculate_expert_dimensions(
            config['total_params'], num_experts=8, d_model=config['d_model']
        )
        
        config.update(expert_config)
        return config


class RWKVExpert(nn.Module):
    """
    Individual RWKV expert for MoE routing.
    
    Uses only the ChannelMix component as the expert since TimeMix
    should be shared across all experts for sequence processing.
    """
    
    def __init__(self, d_model: int, expert_d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.expert_d_ff = expert_d_ff
        
        # Use RWKV ChannelMix as expert
        self.channel_mix = ChannelMix(d_model, expert_d_ff)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expert forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Expert output [batch_size, seq_len, d_model]
        """
        return self.channel_mix(x)
    
    def get_parameter_count(self) -> int:
        """Get the number of parameters in this expert."""
        return sum(p.numel() for p in self.parameters())


class TransformerExpert(nn.Module):
    """
    Individual Transformer expert for MoE routing.
    
    Uses only the FFN component as the expert since attention
    should be shared in the transformer blocks.
    """
    
    def __init__(self, d_model: int, expert_d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.expert_d_ff = expert_d_ff
        
        # Use Transformer FFN as expert
        self.ffn = TransformerFFN(d_model, expert_d_ff, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expert forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Expert output [batch_size, seq_len, d_model]
        """
        return self.ffn(x)
    
    def get_parameter_count(self) -> int:
        """Get the number of parameters in this expert."""
        return sum(p.numel() for p in self.parameters())


class MoERWKVBlock(nn.Module):
    """
    MoE-RWKV Block combining RWKV with expert routing.
    
    Replaces the ChannelMix component with MoE routing to multiple experts.
    Maintains the TimeMix component for sequence processing.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_d_ff: Optional[int] = None,
        capacity_factor: float = 1.25,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Shared TimeMix component (not expert-specific)
        self.time_mixing = TimeMix(d_model)
        
        # MoE router for expert selection
        self.router = MoERouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor
        )
        
        # Create RWKV experts
        if expert_d_ff is None:
            expert_d_ff = 2 * d_model  # Default 2x expansion
        
        self.experts = nn.ModuleList([
            RWKVExpert(d_model, expert_d_ff) 
            for _ in range(num_experts)
        ])
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        MoE-RWKV block forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            state: Optional RWKV state for TimeMix
            return_aux_loss: Whether to return auxiliary losses
            
        Returns:
            output: Processed tensor [batch_size, seq_len, d_model]
            new_state: Updated RWKV state
            aux_losses: MoE auxiliary losses (optional)
        """
        # TimeMix with residual connection (shared across all experts)
        time_mixed, new_state = self.time_mixing(self.ln1(x), state)
        x = x + time_mixed
        
        # MoE expert routing for ChannelMix replacement
        expert_input = self.ln2(x)
        
        # Route to experts
        routing_weights, selected_experts, aux_losses = self.router(
            expert_input, return_aux_loss=return_aux_loss
        )
        
        # Execute experts and combine results
        expert_output = self._execute_experts(
            expert_input, routing_weights, selected_experts
        )
        
        # Final residual connection
        output = x + expert_output
        
        if return_aux_loss:
            return output, new_state, aux_losses
        else:
            return output, new_state
    
    def _execute_experts(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """
        Execute selected experts and combine their outputs.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            routing_weights: Expert weights [batch_size, seq_len, top_k]
            selected_experts: Expert indices [batch_size, seq_len, top_k]
            
        Returns:
            Combined expert output [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx)
            
            if not expert_mask.any():
                continue  # No tokens for this expert
            
            # Get expert weights for tokens assigned to this expert
            expert_weights = torch.where(expert_mask, routing_weights, 0.0)
            
            # Sum weights across top_k dimension to get per-token weights
            token_weights = expert_weights.sum(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
            
            # Only process if expert has non-zero weight
            if token_weights.sum() > 0:
                # Execute expert
                expert_output = self.experts[expert_idx](x)
                
                # Apply expert weights
                weighted_output = expert_output * token_weights
                output = output + weighted_output
        
        return output
    
    def get_expert_statistics(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Get detailed statistics about expert usage.
        
        Args:
            x: Input tensor for analysis
            
        Returns:
            Dictionary of expert statistics
        """
        with torch.no_grad():
            router_stats = self.router.get_routing_statistics(x)
            
            # Add expert parameter counts
            expert_params = [expert.get_parameter_count() for expert in self.experts]
            total_expert_params = sum(expert_params)
            
            stats = {
                **router_stats,
                'expert_parameter_counts': expert_params,
                'total_expert_parameters': total_expert_params,
                'average_parameters_per_expert': total_expert_params / self.num_experts,
                'num_experts': self.num_experts,
                'top_k': self.top_k
            }
            
            return stats


class MoETransformerBlock(nn.Module):
    """
    MoE-Transformer Block combining Transformer attention with expert routing.
    
    Replaces the FFN component with MoE routing to multiple experts.
    Maintains the attention component for all tokens.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_d_ff: Optional[int] = None,
        capacity_factor: float = 1.25,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Shared attention component (not expert-specific)
        from .transformer_block import MultiHeadSelfAttention
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        
        # MoE router for expert selection
        self.router = MoERouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor
        )
        
        # Create Transformer experts
        if expert_d_ff is None:
            expert_d_ff = 4 * d_model  # Standard 4x expansion
        
        self.experts = nn.ModuleList([
            TransformerExpert(d_model, expert_d_ff, dropout) 
            for _ in range(num_experts)
        ])
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        MoE-Transformer block forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            return_aux_loss: Whether to return auxiliary losses
            
        Returns:
            output: Processed tensor [batch_size, seq_len, d_model]
            aux_losses: MoE auxiliary losses (optional)
        """
        # Attention with residual connection (shared)
        attention_input = self.ln1(x)
        attention_output = self.attention(attention_input, mask)
        x = x + self.dropout(attention_output)
        
        # MoE expert routing for FFN replacement
        expert_input = self.ln2(x)
        
        # Route to experts
        routing_weights, selected_experts, aux_losses = self.router(
            expert_input, return_aux_loss=return_aux_loss
        )
        
        # Execute experts and combine results
        expert_output = self._execute_experts(
            expert_input, routing_weights, selected_experts
        )
        
        # Final residual connection
        output = x + self.dropout(expert_output)
        
        if return_aux_loss:
            return output, aux_losses
        else:
            return output
    
    def _execute_experts(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """
        Execute selected experts and combine their outputs.
        
        Same implementation as MoERWKVBlock but for Transformer experts.
        """
        batch_size, seq_len, d_model = x.shape
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx)
            
            if not expert_mask.any():
                continue  # No tokens for this expert
            
            # Get expert weights for tokens assigned to this expert
            expert_weights = torch.where(expert_mask, routing_weights, 0.0)
            
            # Sum weights across top_k dimension to get per-token weights
            token_weights = expert_weights.sum(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
            
            # Only process if expert has non-zero weight
            if token_weights.sum() > 0:
                # Execute expert
                expert_output = self.experts[expert_idx](x)
                
                # Apply expert weights
                weighted_output = expert_output * token_weights
                output = output + weighted_output
        
        return output
    
    def get_expert_statistics(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Get detailed statistics about expert usage.
        
        Args:
            x: Input tensor for analysis
            
        Returns:
            Dictionary of expert statistics
        """
        with torch.no_grad():
            router_stats = self.router.get_routing_statistics(x)
            
            # Add expert parameter counts
            expert_params = [expert.get_parameter_count() for expert in self.experts]
            total_expert_params = sum(expert_params)
            
            stats = {
                **router_stats,
                'expert_parameter_counts': expert_params,
                'total_expert_parameters': total_expert_params,
                'average_parameters_per_expert': total_expert_params / self.num_experts,
                'num_experts': self.num_experts,
                'top_k': self.top_k
            }
            
            return stats
