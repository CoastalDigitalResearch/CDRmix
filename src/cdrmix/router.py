"""
MoE Router Implementation for CDRmix

Implements the mixture-of-experts routing mechanism with:
- Top-k expert selection (k=2)
- Load balancing with auxiliary losses
- Capacity constraints following Lean specifications
- NoOverflow property validation

Mathematical foundation: perExpertCapacity = ⌈ϕ * (N*topK / E)⌉
where ϕ >= 1 is capacity factor, N is tokens, E is experts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class MoERouter(nn.Module):
    """
    MoE Router implementing top-k expert selection with load balancing.
    
    Based on the Lean specification with capacity constraints and NoOverflow property.
    Supports auxiliary losses for load balancing and routing stability.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        dropout: float = 0.1,
        aux_loss_weight: float = 0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        
        # Routing network - learns to route tokens to experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize routing weights
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize routing parameters for stability."""
        # Small initialization to encourage balanced routing initially
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.1)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Route tokens to top-k experts.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            return_aux_loss: Whether to return auxiliary losses
            
        Returns:
            routing_weights: Expert weights [batch_size, seq_len, top_k]
            selected_experts: Expert indices [batch_size, seq_len, top_k] 
            aux_losses: Dictionary of auxiliary losses (optional)
        """
        batch_size, seq_len, d_model = x.shape
        num_tokens = batch_size * seq_len
        
        # Flatten tokens for routing
        x_flat = x.view(-1, d_model)  # [num_tokens, d_model]
        
        # Compute routing logits
        routing_logits = self.gate(x_flat)  # [num_tokens, num_experts]
        routing_logits = self.dropout(routing_logits)
        
        # Apply top-k selection
        top_k_logits, top_k_indices = torch.topk(routing_logits, self.top_k, dim=-1)
        # top_k_logits: [num_tokens, top_k]
        # top_k_indices: [num_tokens, top_k]
        
        # Compute routing weights with softmax over selected experts
        routing_weights = F.softmax(top_k_logits, dim=-1)
        
        # Reshape back to original dimensions
        routing_weights = routing_weights.view(batch_size, seq_len, self.top_k)
        selected_experts = top_k_indices.view(batch_size, seq_len, self.top_k)
        
        # Compute auxiliary losses for load balancing
        aux_losses = None
        if return_aux_loss:
            aux_losses = self._compute_auxiliary_losses(
                routing_logits, top_k_indices, num_tokens
            )
        
        return routing_weights, selected_experts, aux_losses
    
    def _compute_auxiliary_losses(
        self, 
        routing_logits: torch.Tensor,
        selected_experts: torch.Tensor,
        num_tokens: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary losses for load balancing and routing stability.
        
        Args:
            routing_logits: Raw routing logits [num_tokens, num_experts]
            selected_experts: Selected expert indices [num_tokens, top_k]
            num_tokens: Total number of tokens
            
        Returns:
            Dictionary containing auxiliary losses
        """
        aux_losses = {}
        
        # Load balancing loss - encourages uniform expert usage
        load_balance_loss = self._compute_load_balance_loss(
            routing_logits, selected_experts, num_tokens
        )
        aux_losses['load_balance'] = load_balance_loss
        
        # Z-loss - encourages routing confidence
        z_loss = self._compute_z_loss(routing_logits)
        aux_losses['z_loss'] = z_loss
        
        # Capacity overflow detection
        overflow_loss = self._compute_capacity_overflow_loss(
            selected_experts, num_tokens
        )
        aux_losses['overflow'] = overflow_loss
        
        return aux_losses
    
    def _compute_load_balance_loss(
        self,
        routing_logits: torch.Tensor,
        selected_experts: torch.Tensor, 
        num_tokens: int
    ) -> torch.Tensor:
        """
        Compute load balancing loss to encourage uniform expert usage.
        
        Based on the standard MoE load balancing objective.
        """
        # Compute expert selection frequencies
        expert_counts = torch.zeros(self.num_experts, device=routing_logits.device)
        
        # Count how many tokens are routed to each expert
        flat_experts = selected_experts.view(-1)  # [num_tokens * top_k]
        for expert_id in range(self.num_experts):
            expert_counts[expert_id] = (flat_experts == expert_id).float().sum()
        
        # Normalize by total assignments
        total_assignments = num_tokens * self.top_k
        expert_frequencies = expert_counts / (total_assignments + 1e-8)
        
        # Compute routing probabilities (before top-k)
        routing_probs = F.softmax(routing_logits, dim=-1)  # [num_tokens, num_experts]
        avg_routing_probs = routing_probs.mean(dim=0)  # [num_experts]
        
        # Load balance loss: minimize product of frequencies and probabilities
        # This encourages uniform distribution
        load_balance_loss = (expert_frequencies * avg_routing_probs).sum() * self.num_experts
        
        return load_balance_loss
    
    def _compute_z_loss(self, routing_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute z-loss to encourage routing confidence and stability.
        
        Z-loss penalizes large logit values to prevent overconfident routing.
        """
        # Z-loss is the squared L2 norm of logits
        z_loss = (routing_logits ** 2).mean()
        return z_loss
    
    def _compute_capacity_overflow_loss(
        self,
        selected_experts: torch.Tensor,
        num_tokens: int
    ) -> torch.Tensor:
        """
        Compute capacity overflow loss based on Lean specification.
        
        Penalizes routing that violates capacity constraints.
        """
        # Calculate per-expert capacity from Lean spec
        # capacity = ⌈ϕ * (N*topK / E)⌉
        expected_load_per_expert = (num_tokens * self.top_k) / self.num_experts
        per_expert_capacity = math.ceil(self.capacity_factor * expected_load_per_expert)
        
        # Count actual assignments per expert
        expert_loads = torch.zeros(self.num_experts, device=selected_experts.device)
        flat_experts = selected_experts.view(-1)
        
        for expert_id in range(self.num_experts):
            expert_loads[expert_id] = (flat_experts == expert_id).float().sum()
        
        # Compute overflow penalty
        overflow = F.relu(expert_loads - per_expert_capacity)
        overflow_loss = overflow.sum()
        
        return overflow_loss
    
    def get_routing_statistics(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Get detailed routing statistics for monitoring.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Dictionary of routing statistics
        """
        with torch.no_grad():
            routing_weights, selected_experts, aux_losses = self.forward(x)
            
            # Compute statistics
            batch_size, seq_len, _ = x.shape
            num_tokens = batch_size * seq_len
            
            # Expert usage distribution
            flat_experts = selected_experts.view(-1)
            expert_counts = torch.zeros(self.num_experts)
            for expert_id in range(self.num_experts):
                expert_counts[expert_id] = (flat_experts == expert_id).float().sum()
            
            expert_usage = expert_counts / (num_tokens * self.top_k)
            
            # Load balance entropy (higher = more balanced)
            entropy = -(expert_usage * (expert_usage + 1e-8).log()).sum()
            
            # Capacity utilization
            expected_load = (num_tokens * self.top_k) / self.num_experts
            max_load = expert_counts.max().item()
            capacity = math.ceil(self.capacity_factor * expected_load)
            
            stats = {
                'entropy': entropy.item(),
                'max_expert_load': max_load,
                'expected_load': expected_load,
                'capacity_per_expert': capacity,
                'capacity_utilization': max_load / capacity,
                'load_balance_loss': aux_losses['load_balance'].item() if aux_losses else 0.0,
                'z_loss': aux_losses['z_loss'].item() if aux_losses else 0.0,
                'overflow_loss': aux_losses['overflow'].item() if aux_losses else 0.0
            }
            
            return stats
    
    def check_no_overflow_property(
        self, 
        selected_experts: torch.Tensor,
        num_tokens: int
    ) -> bool:
        """
        Check the NoOverflow property from Lean specification.
        
        Returns True if no expert exceeds its capacity.
        """
        # Calculate capacity
        expected_load_per_expert = (num_tokens * self.top_k) / self.num_experts
        capacity = math.ceil(self.capacity_factor * expected_load_per_expert)
        
        # Count actual loads
        flat_experts = selected_experts.view(-1)
        expert_loads = torch.zeros(self.num_experts)
        
        for expert_id in range(self.num_experts):
            expert_loads[expert_id] = (flat_experts == expert_id).float().sum()
        
        # Check if any expert exceeds capacity
        max_load = expert_loads.max().item()
        return max_load <= capacity
