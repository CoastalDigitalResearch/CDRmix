"""
Debug CDRmix Model Creation

Check for issues in model creation process.
"""

import torch
from src.cdrmix.moe_block import ExpertParameterCalculator

def debug_config():
    """Debug configuration creation."""
    print("🔍 Debugging Configuration Creation...")
    
    try:
        config = ExpertParameterCalculator.get_model_scale_config('1b')
        print(f"   1B Config: {config}")
        return True
    except Exception as e:
        print(f"   Error: {e}")
        return False

def debug_scheduler():
    """Debug scheduler creation."""
    print("🔍 Debugging Scheduler Creation...")
    
    try:
        from src.cdrmix.rwkv_x_architecture import RWKVXScheduler
        scheduler = RWKVXScheduler(tx_ratio=0.25)
        schedule = scheduler.schedule_hybrid(24, k=4)
        print(f"   Schedule length: {len(schedule)}")
        return True
    except Exception as e:
        print(f"   Error: {e}")
        return False

def debug_moe_blocks():
    """Debug MoE block creation."""
    print("🔍 Debugging MoE Block Creation...")
    
    try:
        from src.cdrmix.moe_block import MoERWKVBlock
        block = MoERWKVBlock(d_model=512, num_experts=4, top_k=2, expert_d_ff=1024)
        print(f"   MoE RWKV Block created successfully")
        return True
    except Exception as e:
        print(f"   Error: {e}")
        return False

def main():
    """Run all debug tests."""
    print("🚨 CDRmix Debug Session")
    print("=" * 40)
    
    tests = [
        ("Configuration", debug_config),
        ("Scheduler", debug_scheduler), 
        ("MoE Blocks", debug_moe_blocks)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        result = test_func()
        results.append((name, result))
    
    print(f"\n📊 Debug Results:")
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"   {name}: {status}")
    
    if all(result for _, result in results):
        print(f"\n✅ All components working - issue might be in model integration")
    else:
        print(f"\n❌ Found issues in basic components")

if __name__ == "__main__":
    main()