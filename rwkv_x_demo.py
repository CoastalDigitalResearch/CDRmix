"""
RWKV-X Architecture Demonstration

Shows the hybrid RWKV-X architecture in action with different scheduling patterns.
Demonstrates the 25% transformer / 75% RWKV block distribution.
"""

import torch
from src.cdrmix.rwkv_x_architecture import RWKVXArchitecture, ScheduleType, RWKVXScheduler

def demonstrate_rwkv_x():
    """Demonstrate RWKV-X architecture functionality."""
    print("🚀 RWKV-X Hybrid Architecture Demonstration")
    print("=" * 50)
    
    # Architecture parameters
    num_layers = 24
    d_model = 512
    batch_size, seq_len = 2, 128
    
    print(f"Model Configuration:")
    print(f"  - Layers: {num_layers}")
    print(f"  - Model Dimension: {d_model}")
    print(f"  - Target: 25% Transformer, 75% RWKV")
    print()
    
    # Demonstrate different scheduling patterns
    schedule_types = [
        (ScheduleType.TOP_OF_X, "Top-of-X (Transformers at end)"),
        (ScheduleType.INTERLEAVE_X, "Interleave-X (Transformers distributed)"), 
        (ScheduleType.HYBRID, "Hybrid (Mixed strategy)")
    ]
    
    for schedule_type, description in schedule_types:
        print(f"📋 {description}")
        print("-" * 40)
        
        # Create architecture
        arch = RWKVXArchitecture(
            num_layers=num_layers,
            d_model=d_model,
            schedule_type=schedule_type
        )
        
        # Get schedule information
        schedule_info = arch.get_schedule_info()
        print(f"  Schedule Pattern: {schedule_info['schedule']}")
        print(f"  RWKV Blocks: {schedule_info['num_rwkv']} ({schedule_info['rwkv_ratio']:.1%})")
        print(f"  Transformer Blocks: {schedule_info['num_transformer']} ({schedule_info['tx_ratio']:.1%})")
        
        # Test forward pass
        x = torch.randn(batch_size, seq_len, d_model)
        
        with torch.no_grad():
            output = arch(x)
            print(f"  Forward Pass: {x.shape} → {output.shape} ✓")
            
            # Get detailed analysis
            analysis = arch.forward_with_layer_analysis(x)
            complexity = analysis["schedule_info"]["complexity_analysis"]
            
            print(f"  Complexity Analysis:")
            print(f"    - Linear Component: {complexity['linear_component']:.1%}")
            print(f"    - Quadratic Component: {complexity['quadratic_component']:.1%}")
            print(f"    - Total Cost: {complexity['total_cost']:,} FLOPs")
        
        print()
    
    # Demonstrate scheduler directly
    print("⚙️  RWKV-X Scheduler Analysis")
    print("-" * 40)
    
    scheduler = RWKVXScheduler(tx_ratio=0.25)
    
    # Compare different layer counts
    layer_counts = [16, 24, 32, 48]
    
    for layers in layer_counts:
        top_schedule = scheduler.schedule_top_of_x(layers)
        interleave_schedule = scheduler.schedule_interleave_x(layers, k=4)
        hybrid_schedule = scheduler.schedule_hybrid(layers, k=8)
        
        print(f"  {layers} Layers:")
        print(f"    Top-of-X Tx: {top_schedule.count(BlockType.TRANSFORMER)}/{layers} = {top_schedule.count(BlockType.TRANSFORMER)/layers:.1%}")
        print(f"    Interleave Tx: {interleave_schedule.count(BlockType.TRANSFORMER)}/{layers} = {interleave_schedule.count(BlockType.TRANSFORMER)/layers:.1%}")
        print(f"    Hybrid Tx: {hybrid_schedule.count(BlockType.TRANSFORMER)}/{layers} = {hybrid_schedule.count(BlockType.TRANSFORMER)/layers:.1%}")
    
    print()
    print("✅ RWKV-X Architecture Ready for MoE Integration!")


if __name__ == "__main__":
    # Fix import for demo
    import sys
    sys.path.append('.')
    from src.cdrmix.rwkv_x_architecture import BlockType
    
    demonstrate_rwkv_x()