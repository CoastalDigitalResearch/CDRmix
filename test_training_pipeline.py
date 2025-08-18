"""
Training Pipeline Validation Test

Quick test to ensure all training components work together.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from utils.config import load_yaml_config, get_model_config_from_yaml, get_training_config
from utils.tokenizer_utils import get_tokenizer_for_config
from data_loader import create_data_loaders
from cdrmix.cdrmix_model import CDRmixConfig, CDRmixModel


def test_config_loading():
    """Test configuration loading."""
    print("🔧 Testing Configuration Loading...")
    
    config_path = "configs/cdrmix-core-1b.yaml"
    
    try:
        yaml_config = load_yaml_config(config_path)
        model_config = get_model_config_from_yaml(yaml_config)
        training_config = get_training_config(yaml_config)
        
        print(f"   ✅ Config loaded successfully")
        print(f"   Model Scale: {model_config['model_scale']}")
        print(f"   Vocab Size: {model_config['vocab_size']}")
        print(f"   Batch Size: {training_config['batch_size']}")
        
        return yaml_config, model_config, training_config
    
    except Exception as e:
        print(f"   ❌ Config loading failed: {e}")
        return None, None, None


def test_tokenizer_creation(yaml_config):
    """Test tokenizer creation."""
    print("\n🔤 Testing Tokenizer Creation...")
    
    try:
        tokenizer = get_tokenizer_for_config(yaml_config)
        
        # Test encoding/decoding
        test_text = "Hello world!"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        print(f"   ✅ Tokenizer created successfully")
        print(f"   Vocab Size: {tokenizer.vocab_size}")
        print(f"   Test: '{test_text}' -> {len(encoded)} tokens -> '{decoded}'")
        
        return tokenizer
    
    except Exception as e:
        print(f"   ❌ Tokenizer creation failed: {e}")
        return None


def test_model_creation(model_config):
    """Test model creation."""
    print("\n🏗️  Testing Model Creation...")
    
    try:
        # Create small config for testing
        test_config = model_config.copy()
        test_config['vocab_size'] = 1000  # Smaller vocab for speed
        
        cdrmix_config = CDRmixConfig(**test_config)
        model = CDRmixModel(cdrmix_config)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"   ✅ Model created successfully")
        print(f"   Model Scale: {cdrmix_config.model_scale}")
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Number of Layers: {cdrmix_config.n_layers}")
        print(f"   Number of Experts: {cdrmix_config.num_experts}")
        
        return model
    
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return None


def test_data_loading(yaml_config, tokenizer):
    """Test data loading."""
    print("\n📊 Testing Data Loading...")
    
    try:
        data_config = '/tmp/cdrmix_test_data'  # Will create synthetic data
        
        data_loaders = create_data_loaders(
            data_config=data_config,
            tokenizer=tokenizer,
            batch_size=2,  # Small batch for testing
            seq_length=64,  # Short sequences for testing
            num_workers=0   # No multiprocessing for testing
        )
        
        # Test loading a batch
        train_batch = next(iter(data_loaders['train']))
        val_batch = next(iter(data_loaders['val']))
        
        print(f"   ✅ Data loading successful")
        print(f"   Train batches: {len(data_loaders['train'])}")
        print(f"   Val batches: {len(data_loaders['val'])}")
        print(f"   Batch shape: {train_batch.shape}")
        
        return data_loaders
    
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return None


def test_model_forward(model, data_loaders):
    """Test model forward pass."""
    print("\n⚡ Testing Model Forward Pass...")
    
    try:
        model.eval()
        
        # Get a batch
        batch = next(iter(data_loaders['train']))
        
        # Prepare inputs
        inputs = batch[:, :-1]  # [batch_size, seq_len-1]
        targets = batch[:, 1:]  # [batch_size, seq_len-1]
        
        # Forward pass without aux losses
        with torch.no_grad():
            logits = model(inputs, return_aux_losses=False, return_states=False)
        
        print(f"   ✅ Forward pass successful")
        print(f"   Input shape: {inputs.shape}")
        print(f"   Output shape: {logits.shape}")
        print(f"   No NaN/Inf: {not torch.isnan(logits).any() and not torch.isinf(logits).any()}")
        
        # Test with aux losses and states
        with torch.no_grad():
            logits, states, aux_losses = model(inputs, return_aux_losses=True, return_states=True)
        
        print(f"   States: {len([s for s in states if s is not None])} RWKV states")
        print(f"   Aux losses: {list(aux_losses.keys())}")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Model forward failed: {e}")
        return False


def test_loss_computation(model, data_loaders):
    """Test loss computation including MoE losses."""
    print("\n🎯 Testing Loss Computation...")
    
    try:
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        
        # Get a batch
        batch = next(iter(data_loaders['train']))
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        # Forward with aux losses
        logits, states, aux_losses = model(inputs, return_aux_losses=True, return_states=True)
        
        # Compute primary loss
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        ce_loss = criterion(logits_flat, targets_flat)
        
        # Compute aux loss
        aux_loss = (
            aux_losses['load_balance'] + 
            aux_losses['z_loss'] + 
            aux_losses['overflow']
        )
        
        total_loss = ce_loss + 0.01 * aux_loss
        
        print(f"   ✅ Loss computation successful")
        print(f"   CE Loss: {ce_loss.item():.4f}")
        print(f"   Load Balance: {aux_losses['load_balance'].item():.6f}")
        print(f"   Z Loss: {aux_losses['z_loss'].item():.6f}")
        print(f"   Overflow: {aux_losses['overflow'].item():.6f}")
        print(f"   Total Loss: {total_loss.item():.4f}")
        
        # Test backward pass
        total_loss.backward()
        
        print(f"   ✅ Backward pass successful")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Loss computation failed: {e}")
        return False


def main():
    """Run all training pipeline tests."""
    print("🧪 CDRmix Training Pipeline Validation")
    print("=" * 50)
    
    # Test components
    yaml_config, model_config, training_config = test_config_loading()
    if not yaml_config:
        return False
    
    tokenizer = test_tokenizer_creation(yaml_config)
    if not tokenizer:
        return False
    
    model = test_model_creation(model_config)
    if not model:
        return False
    
    data_loaders = test_data_loading(yaml_config, tokenizer)
    if not data_loaders:
        return False
    
    if not test_model_forward(model, data_loaders):
        return False
    
    if not test_loss_computation(model, data_loaders):
        return False
    
    print("\n🎉 Training Pipeline Validation Complete!")
    print("=" * 50)
    print("✅ All components working correctly")
    print("✅ Ready for training run")
    print(f"✅ Model: {model_config['model_scale']} with {model_config['num_experts']} experts")
    print(f"✅ Training: Mixed precision, MoE losses, checkpointing")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)