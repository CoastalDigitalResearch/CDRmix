"""
RWKV Block Implementation for CDRmix

Implements the RWKV (Receptance Weighted Key Value) architecture with:
- TimeMix: Linear O(n*d) token mixing vs O(n²*d) quadratic mechanisms
- ChannelMix: Position-wise FFN with special gating
- Streaming capability for incremental processing

Mathematical formulation: WKV = Σ(exp(W + K) · V) / Σ(exp(W + K))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TimeMix(nn.Module):
    """
    RWKV TimeMix component implementing linear-complexity token mixing.
    
    This replaces traditional quadratic mechanisms with O(n*d) complexity instead of O(n²*d).
    Uses the RWKV formulation: WKV = Σ(exp(W + K) · V) / Σ(exp(W + K))
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # RWKV projection matrices - core RWKV components
        self.receptance = nn.Linear(d_model, d_model, bias=False)  # receptance
        self.weight = nn.Linear(d_model, d_model, bias=False)      # weight  
        self.key = nn.Linear(d_model, d_model, bias=False)         # key
        self.value = nn.Linear(d_model, d_model, bias=False)       # value
        
        # Output projection
        self.output = nn.Linear(d_model, d_model, bias=False)
        
        # Time decay parameter for recurrent computation
        self.time_decay = nn.Parameter(torch.zeros(d_model))
        self.time_first = nn.Parameter(torch.zeros(d_model))
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """
Initialize parameters following RWKV best practices.
        """
        # Initialize projections with Xavier uniform
        for module in [self.receptance, self.weight, self.key, self.value, self.output]:
            nn.init.xavier_uniform_(module.weight)
        
        # Initialize time parameters
        nn.init.uniform_(self.time_decay, -1.0, 0.0)  # Negative for decay
        nn.init.uniform_(self.time_first, -1.0, 1.0)
    
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass implementing RWKV TimeMix computation.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            state: Optional recurrent state for streaming
            
        Returns:
            output: Mixed tokens [batch_size, seq_len, d_model]
            new_state: Updated recurrent state
        """
        batch_size, seq_len, d_model = x.shape
        
        # Extract RWKV components
        r = torch.sigmoid(self.receptance(x))  # Receptance [0,1]
        w = -torch.exp(self.weight(x))         # Weight (negative for decay)
        k = self.key(x)                        # Key
        v = self.value(x)                      # Value
        
        # Implement RWKV recurrent computation with linear complexity
        initial_state = state is None
        if initial_state:
            # Initialize state with proper dimensions [batch_size, 2, d_model]
            # state[0] = numerator_state, state[1] = denominator_state
            state = torch.zeros(batch_size, 2, d_model, device=x.device, dtype=x.dtype)
        
        outputs = []
        current_num_state = state[:, 0, :]  # Numerator state
        current_den_state = state[:, 1, :]  # Denominator state
        
        for t in range(seq_len):
            # Current timestep inputs
            rt = r[:, t, :]  # [batch_size, d_model]
            wt = w[:, t, :]  # [batch_size, d_model]
            kt = k[:, t, :]  # [batch_size, d_model]
            vt = v[:, t, :]  # [batch_size, d_model]
            
            # RWKV computation: weighted key-value with time decay
            exp_wk = torch.exp(wt + kt)
            
            if t == 0 and initial_state:
                # First timestep initialization
                numerator = exp_wk * vt
                denominator = exp_wk
            else:
                # Use accumulated state with time decay
                decay = torch.exp(self.time_decay)
                numerator = decay * current_num_state + exp_wk * vt
                denominator = decay * current_den_state + exp_wk
            
            # Compute WKV with numerical stability
            wkv = numerator / (denominator + 1e-8)
            
            # Apply receptance gating
            output_t = rt * wkv
            outputs.append(output_t)
            
            # Update states for next timestep
            current_num_state = numerator
            current_den_state = denominator
        
        # Stack outputs and apply final projection
        output = torch.stack(outputs, dim=1)  # [batch_size, seq_len, d_model]
        output = self.output(output)
        
        # Create final state
        final_state = torch.stack([current_num_state, current_den_state], dim=1)
        
        return output, final_state


class ChannelMix(nn.Module):
    """
    RWKV ChannelMix component implementing position-wise FFN with special gating.
    
    This is similar to a standard FFN but with RWKV-style gating mechanisms.
    """
    
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model  # Standard 4x expansion
        
        # RWKV-style projections with gating
        self.key_proj = nn.Linear(d_model, self.d_ff, bias=False)
        self.value_proj = nn.Linear(self.d_ff, d_model, bias=False)
        self.receptance_proj = nn.Linear(d_model, d_model, bias=False)
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters with appropriate scaling."""
        # Xavier initialization for all projections
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.receptance_proj.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing RWKV ChannelMix computation.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # RWKV ChannelMix with squared ReLU activation and receptance gating
        k = self.key_proj(x)
        k = torch.relu(k) ** 2  # Squared ReLU for better gradient flow
        v = self.value_proj(k)
        
        # Apply receptance gating
        r = torch.sigmoid(self.receptance_proj(x))
        output = r * v
        
        return output


class RWKVBlock(nn.Module):
    """
    Complete RWKV block combining TimeMix and ChannelMix components.
    
    Implements the full RWKV architecture with:
    - Linear complexity O(n*d) token mixing
    - Position-wise channel mixing with gating
    - Layer normalization for stability
    - Residual connections
    """
    
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        
        # Core RWKV components
        self.time_mixing = TimeMix(d_model)
        self.channel_mixing = ChannelMix(d_model, d_ff)
        
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through complete RWKV block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            state: Optional recurrent state for streaming
            
        Returns:
            output: Processed tensor [batch_size, seq_len, d_model]
            new_state: Updated recurrent state
        """
        # TimeMix with residual connection and layer norm
        time_mixed, new_state = self.time_mixing(self.ln1(x), state)
        x = x + time_mixed
        
        # ChannelMix with residual connection and layer norm
        channel_mixed = self.channel_mixing(self.ln2(x))
        x = x + channel_mixed
        
        return x, new_state
    
    def forward_incremental(self, token: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single token incrementally for streaming applications.
        
        Args:
            token: Single token [batch_size, 1, d_model]
            state: Recurrent state from previous tokens
            
        Returns:
            output: Processed token [batch_size, 1, d_model]
            new_state: Updated recurrent state
        """
        return self.forward(token, state)
