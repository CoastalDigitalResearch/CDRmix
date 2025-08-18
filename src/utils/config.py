"""
Configuration utilities for CDRmix training.

Handles YAML configuration loading and validation for model training.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        ValueError: If required fields are missing
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")
    
    # Validate configuration structure
    _validate_config(config, config_path)
    
    return config


def _validate_config(config: Dict[str, Any], config_path: Path) -> None:
    """Validate configuration has required fields."""
    required_sections = ['model']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in {config_path}")
    
    # Validate model section
    model_config = config['model']
    required_model_fields = ['name', 'vocab_size', 'd_model', 'n_layers']
    
    for field in required_model_fields:
        if field not in model_config:
            raise ValueError(f"Missing required model field '{field}' in {config_path}")


def get_model_config_from_yaml(yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert YAML config to CDRmix model configuration.
    
    Maps YAML configuration structure to CDRmixConfig parameters.
    """
    yaml_model = yaml_config['model']
    
    # Extract model scale from name
    model_name = yaml_model['name'].lower()
    if '1b' in model_name:
        model_scale = '1b'
    elif '4b' in model_name:
        model_scale = '4b'
    elif '40b' in model_name:
        model_scale = '40b'
    elif '200b' in model_name:
        model_scale = '200b'
    else:
        model_scale = '1b'  # Default fallback
    
    # Base configuration
    model_config = {
        'model_scale': model_scale,
        'vocab_size': yaml_model['vocab_size']
    }
    
    # MoE configuration
    if 'moe' in yaml_model and yaml_model['moe'].get('enabled', False):
        moe_config = yaml_model['moe']
        model_config.update({
            'num_experts': moe_config.get('experts', 8),
            'top_k': moe_config.get('top_k', 2),
            'capacity_factor': moe_config.get('capacity_factor', 1.25)
        })
    
    # RWKV-X schedule configuration
    if 'schedule' in yaml_model:
        schedule_config = yaml_model['schedule']
        model_config.update({
            'transformer_ratio': schedule_config.get('transformer_pct', 0.25)
        })
    
    # Training configuration
    if 'training' in yaml_config:
        training_config = yaml_config['training']
        model_config.update({
            'dropout': training_config.get('dropout', 0.1)
        })
    
    # Embedding configuration
    if 'readout' in yaml_model:
        readout_config = yaml_model['readout']
        model_config.update({
            'tie_embeddings': readout_config.get('tie_embeddings', True)
        })
    
    return model_config


def get_training_config(yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training-specific configuration."""
    default_training = {
        'batch_size': 4,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'gradient_clip_val': 1.0,
        'warmup_steps': 2000,
        'max_steps': 100000,
        'eval_interval': 1000,
        'save_interval': 5000,
        'dtype': 'bfloat16'
    }
    
    if 'training' not in yaml_config:
        return default_training
    
    training_config = yaml_config['training'].copy()
    
    # Add defaults for missing fields
    for key, default_value in default_training.items():
        if key not in training_config:
            training_config[key] = default_value
    
    # Handle MoE auxiliary losses
    if 'losses' in training_config and 'moe_aux' in training_config['losses']:
        training_config['use_moe_aux_loss'] = True
        training_config['aux_loss_weight'] = training_config.get('aux_loss_weight', 0.01)
    else:
        training_config['use_moe_aux_loss'] = False
    
    return training_config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, indent=2)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations, with override taking precedence."""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged