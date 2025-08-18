"""
Transformer Block Implementation for CDRmix RWKV-X Architecture

Implements traditional transformer blocks that are interleaved with RWKV blocks
in the RWKV-X pattern (25% transformer, 75% RWKV).

This provides the O(n²*d) quadratic attention component that complements 
the linear O(n*d) RWKV blocks for enhanced modeling capacity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) implementation.
    
    Provides the quadratic O(n²*d) attention mechanism that complements
    RWKV's linear complexity for modeling long-range dependencies.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters with scaled Xavier initialization."""
        # Scale initialization for stability with multiple attention heads
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1.0 / math.sqrt(3))
        nn.init.xavier_uniform_(self.output_proj.weight)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Multi-head self-attention forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V projections
        qkv = self.qkv_proj(x)  # [batch_size, seq_len, 3*d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, n_heads, seq_len, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # [batch_size, n_heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, v)
        # [batch_size, n_heads, seq_len, d_head]
        
        # Concatenate heads and project
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(batch_size, seq_len, d_model)
        
        output = self.output_proj(attention_output)
        return output


class TransformerFFN(nn.Module):
    """
    Transformer Feed-Forward Network with GELU activation.
    
    Implements the position-wise feed-forward network used in standard transformers.
    """
    
    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model  # Standard 4x expansion
        
        self.linear1 = nn.Linear(d_model, self.d_ff, bias=False)
        self.linear2 = nn.Linear(self.d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize FFN parameters."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FFN forward pass with GELU activation.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Two-layer MLP with GELU activation
        x = self.linear1(x)
        x = F.gelu(x)  # GELU activation as specified
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Standard transformer block for RWKV-X interleaving.
    
    Combines multi-head self-attention and feed-forward network with
    layer normalization and residual connections.
    
    This provides the O(n²*d) quadratic modeling capacity in the RWKV-X
    hybrid architecture.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Core transformer components
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = TransformerFFN(d_model, d_ff, dropout)
        
        # Layer normalization (pre-norm architecture)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """
        Transformer block forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Multi-head self-attention with pre-norm and residual
        attention_input = self.ln1(x)
        attention_output = self.attention(attention_input, mask)
        x = x + self.dropout(attention_output)
        
        # Feed-forward network with pre-norm and residual
        ffn_input = self.ln2(x)
        ffn_output = self.ffn(ffn_input)
        x = x + self.dropout(ffn_output)
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get attention weights for analysis/visualization.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Attention weights [batch_size, n_heads, seq_len, seq_len]
        """
        # This is a simplified version - would need to modify MHSA to return weights
        with torch.no_grad():
            attention_input = self.ln1(x)
            batch_size, seq_len, d_model = attention_input.shape
            
            # Recompute attention weights without modifying the forward pass
            qkv = self.attention.qkv_proj(attention_input)
            qkv = qkv.reshape(batch_size, seq_len, 3, self.attention.n_heads, self.attention.d_head)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attention.scale
            
            if mask is not None:
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
            
            attention_weights = F.softmax(attention_scores, dim=-1)
            return attention_weights