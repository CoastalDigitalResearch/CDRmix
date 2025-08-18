"""
CDRmix Model - Complete Integration

Combines RWKV-X architecture with MoE routing for the full CDRmix system.
Supports all model scales (1B, 4B, 40B, 200B) with 8 experts each.

Architecture:
- RWKV-X scheduling (Top-of-X and Interleave-X patterns)
- MoE routing with top-k=2 expert selection
- Expert parameter scaling across model sizes
- Streaming-compatible RWKV blocks with state management
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union

from .rwkv_x_architecture import RWKVXScheduler, ScheduleType
from .moe_block import MoERWKVBlock, MoETransformerBlock, ExpertParameterCalculator
from .router import MoERouter


class CDRmixConfig:
    """Configuration for CDRmix model across different scales."""
    
    def __init__(
        self,
        model_scale: str = '1b',
        vocab_size: int = 50272,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        dropout: float = 0.1,
        # RWKV-X configuration
        transformer_ratio: float = 0.25,
        top_of_x_ratio: float = 0.5,  # 50% of transformer blocks at top
        # Training configuration
        tie_embeddings: bool = True,
        use_checkpoint: bool = False
    ):
        # Get base model configuration
        base_config = ExpertParameterCalculator.get_model_scale_config(model_scale)
        
        # Model architecture parameters
        self.model_scale = model_scale
        self.vocab_size = vocab_size
        self.d_model = base_config['d_model']
        self.n_layers = base_config['n_layers']
        self.n_heads = base_config['n_heads']
        
        # MoE parameters
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.expert_d_ff = base_config['expert_d_ff']
        
        # RWKV-X architecture parameters
        self.transformer_ratio = transformer_ratio
        self.top_of_x_ratio = top_of_x_ratio
        self.interleave_ratio = 1.0 - top_of_x_ratio
        
        # Compute block distribution
        self.total_transformer_blocks = int(self.n_layers * self.transformer_ratio)
        self.top_of_x_blocks = int(self.total_transformer_blocks * self.top_of_x_ratio)
        self.interleave_blocks = self.total_transformer_blocks - self.top_of_x_blocks
        self.rwkv_blocks = self.n_layers - self.total_transformer_blocks
        
        # Training parameters
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings
        self.use_checkpoint = use_checkpoint
        
        # Expert parameter validation
        self.target_expert_params = base_config['target_params_per_expert']
        self.actual_expert_params = base_config['actual_params_per_expert']
    
    def create_scheduler(self) -> RWKVXScheduler:
        """Create RWKV-X scheduler for this configuration."""
        return RWKVXScheduler(tx_ratio=self.transformer_ratio)


class CDRmixModel(nn.Module):
    """
    Complete CDRmix model with MoE-enhanced RWKV-X architecture.
    
    Integrates:
    - RWKV-X scheduling with transformer and RWKV blocks
    - MoE routing for both RWKV and Transformer experts
    - Parameter scaling across different model sizes
    - Streaming computation compatibility
    """
    
    def __init__(self, config: CDRmixConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        
        # Core architecture components
        self._build_architecture()
        
        # Output projection
        if config.tie_embeddings:
            # Share embedding weights with output projection
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.embeddings.weight
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Layer normalization for final output
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _build_architecture(self):
        """Build the complete MoE-enhanced RWKV-X architecture."""
        # Create RWKV-X scheduler
        scheduler = self.config.create_scheduler()
        
        # Build layer sequence based on RWKV-X pattern
        self.layers = nn.ModuleList()
        layer_types = self._compute_layer_schedule(scheduler)
        
        for i, layer_type in enumerate(layer_types):
            if layer_type == 'transformer':
                # MoE-enhanced Transformer block
                layer = MoETransformerBlock(
                    d_model=self.config.d_model,
                    n_heads=self.config.n_heads,
                    num_experts=self.config.num_experts,
                    top_k=self.config.top_k,
                    expert_d_ff=self.config.expert_d_ff,
                    capacity_factor=self.config.capacity_factor,
                    dropout=self.config.dropout
                )
            else:  # 'rwkv'
                # MoE-enhanced RWKV block
                layer = MoERWKVBlock(
                    d_model=self.config.d_model,
                    num_experts=self.config.num_experts,
                    top_k=self.config.top_k,
                    expert_d_ff=self.config.expert_d_ff,
                    capacity_factor=self.config.capacity_factor,
                    dropout=self.config.dropout
                )
            
            self.layers.append(layer)
    
    def _compute_layer_schedule(self, scheduler: RWKVXScheduler) -> List[str]:
        """Compute the layer schedule following RWKV-X patterns."""
        # Use hybrid schedule combining top-of-x and interleave-x
        schedule_blocks = scheduler.schedule_hybrid(
            num_layers=self.config.n_layers,
            k=4  # Standard interleaving frequency
        )
        
        # Convert block types to strings
        schedule = []
        for block_type in schedule_blocks:
            if block_type.value == 'transformer':
                schedule.append('transformer')
            else:
                schedule.append('rwkv')
        
        return schedule
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rwkv_states: Optional[List[torch.Tensor]] = None,
        return_aux_losses: bool = True,
        return_states: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass through CDRmix model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask for transformer blocks
            rwkv_states: Optional RWKV states for streaming
            return_aux_losses: Whether to return MoE auxiliary losses
            return_states: Whether to return RWKV states
            
        Returns:
            logits: Language modeling logits [batch_size, seq_len, vocab_size]
            states: RWKV states for streaming (optional)
            aux_losses: MoE auxiliary losses (optional)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.embeddings(input_ids)  # [batch_size, seq_len, d_model]
        
        # Initialize states if not provided
        if rwkv_states is None:
            rwkv_states = [None] * len(self.layers)
        
        # Process through layers
        new_states = []
        total_aux_losses = {
            'load_balance': torch.tensor(0.0, device=x.device),
            'z_loss': torch.tensor(0.0, device=x.device),
            'overflow': torch.tensor(0.0, device=x.device)
        }
        aux_loss_count = 0
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, MoERWKVBlock):
                # RWKV block with state
                if return_aux_losses:
                    x, new_state, aux_losses = layer(
                        x, state=rwkv_states[i], return_aux_loss=True
                    )
                    # Accumulate auxiliary losses
                    for key, loss in aux_losses.items():
                        total_aux_losses[key] += loss
                    aux_loss_count += 1
                else:
                    x, new_state = layer(
                        x, state=rwkv_states[i], return_aux_loss=False
                    )
                
                new_states.append(new_state)
                
            else:  # MoETransformerBlock
                # Transformer block
                if return_aux_losses:
                    x, aux_losses = layer(
                        x, mask=attention_mask, return_aux_loss=True
                    )
                    # Accumulate auxiliary losses
                    for key, loss in aux_losses.items():
                        total_aux_losses[key] += loss
                    aux_loss_count += 1
                else:
                    x = layer(
                        x, mask=attention_mask, return_aux_loss=False
                    )
                
                new_states.append(None)  # No state for transformer blocks
        
        # Final layer normalization
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
        
        # Prepare return values
        result = [logits]
        
        if return_states:
            result.append(new_states)
        
        if return_aux_losses and aux_loss_count > 0:
            # Average auxiliary losses across MoE blocks
            avg_aux_losses = {
                key: loss / aux_loss_count for key, loss in total_aux_losses.items()
            }
            result.append(avg_aux_losses)
        
        if len(result) == 1:
            return logits
        else:
            return tuple(result)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        Generate text using the CDRmix model.
        
        Supports streaming generation with RWKV state management.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generation state
        generated = input_ids.clone()
        states = None
        
        # Generation loop
        for _ in range(max_length - input_ids.shape[1]):
            # Get next token logits
            with torch.no_grad():
                if states is None:
                    # First forward pass
                    logits, states = self.forward(
                        generated, return_aux_losses=False, return_states=True
                    )
                    logits = logits[:, -1, :]  # Last token logits
                else:
                    # Streaming forward pass with state
                    last_token = generated[:, -1:]
                    logits, states = self.forward(
                        last_token, rwkv_states=states, 
                        return_aux_losses=False, return_states=True
                    )
                    logits = logits[:, -1, :]
            
            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, _ = torch.topk(logits, top_k)
                logits[logits < top_k_logits[:, -1:]] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits.scatter_(-1, indices_to_remove.unsqueeze(-1), float('-inf'))
            
            # Sample next token
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Check for end of generation
            if (next_token == pad_token_id).all():
                break
        
        return generated
    
    def get_model_statistics(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """Get comprehensive model statistics including MoE routing."""
        stats = {
            'model_scale': self.config.model_scale,
            'total_layers': self.config.n_layers,
            'rwkv_layers': self.config.rwkv_blocks,
            'transformer_layers': self.config.total_transformer_blocks,
            'top_of_x_layers': self.config.top_of_x_blocks,
            'interleave_layers': self.config.interleave_blocks,
            'num_experts': self.config.num_experts,
            'expert_top_k': self.config.top_k,
            'target_expert_params': self.config.target_expert_params,
            'actual_expert_params': self.config.actual_expert_params,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        
        # Get MoE routing statistics
        with torch.no_grad():
            x = self.embeddings(input_ids)
            
            moe_stats = []
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'get_expert_statistics'):
                    layer_stats = layer.get_expert_statistics(x)
                    layer_stats['layer_index'] = i
                    layer_stats['layer_type'] = 'rwkv' if isinstance(layer, MoERWKVBlock) else 'transformer'
                    moe_stats.append(layer_stats)
                    
                    # Forward through layer for next iteration
                    if isinstance(layer, MoERWKVBlock):
                        x, _ = layer(x, return_aux_loss=False)
                    else:
                        x = layer(x, return_aux_loss=False)
            
            stats['moe_layer_statistics'] = moe_stats
        
        return stats


# Factory functions for different model scales
def create_cdrmix_1b(vocab_size: int = 50272, **kwargs) -> CDRmixModel:
    """Create CDRmix 1B parameter model."""
    config = CDRmixConfig(model_scale='1b', vocab_size=vocab_size, **kwargs)
    return CDRmixModel(config)


def create_cdrmix_4b(vocab_size: int = 50272, **kwargs) -> CDRmixModel:
    """Create CDRmix 4B parameter model."""
    config = CDRmixConfig(model_scale='4b', vocab_size=vocab_size, **kwargs)
    return CDRmixModel(config)


def create_cdrmix_40b(vocab_size: int = 50272, **kwargs) -> CDRmixModel:
    """Create CDRmix 40B parameter model."""
    config = CDRmixConfig(model_scale='40b', vocab_size=vocab_size, **kwargs)
    return CDRmixModel(config)


def create_cdrmix_200b(vocab_size: int = 50272, **kwargs) -> CDRmixModel:
    """Create CDRmix 200B parameter model."""
    config = CDRmixConfig(model_scale='200b', vocab_size=vocab_size, **kwargs)
    return CDRmixModel(config)


# Legacy compatibility
class RWKVMoEModel(CDRmixModel):
    """Legacy alias for CDRmixModel."""
    pass
