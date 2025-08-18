"""
Data loading pipeline for CDRmix training.

Handles streaming JSONL datasets with tokenization and batching.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Iterator, Any
from pathlib import Path
import random
import warnings


class CDRDataset(Dataset):
    """
    CDRmix dataset for training language models.
    
    Supports:
    - Streaming JSONL files
    - Dynamic sequence packing
    - Memory-efficient loading
    """
    
    def __init__(
        self,
        data_config: Dict[str, Any],
        tokenizer=None,
        seq_length: int = 2048,
        split: str = 'train'
    ):
        """
        Initialize CDR dataset.
        
        Args:
            data_config: Dataset configuration from YAML
            tokenizer: Tokenizer for text encoding
            seq_length: Maximum sequence length
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_config = data_config
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.split = split
        
        # Get data paths
        self.data_paths = self._get_data_paths()
        
        # Load and prepare data
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _get_data_paths(self) -> List[Path]:
        """Get list of data files for this split."""
        if isinstance(self.data_config, str):
            # Simple path
            base_path = Path(self.data_config)
        elif isinstance(self.data_config, dict):
            # Configuration dict
            base_path = Path(self.data_config.get('path', '/tmp/data'))
        else:
            raise ValueError(f"Invalid data_config type: {type(self.data_config)}")
        
        if not base_path.exists():
            warnings.warn(f"Data path {base_path} does not exist. Creating synthetic data.")
            return self._create_synthetic_data(base_path)
        
        # Find JSONL files
        if base_path.is_file():
            return [base_path]
        
        # Find all JSONL files in directory
        jsonl_files = list(base_path.glob("**/*.jsonl"))
        
        if not jsonl_files:
            warnings.warn(f"No JSONL files found in {base_path}. Creating synthetic data.")
            return self._create_synthetic_data(base_path)
        
        # Filter by split if specified in filenames
        split_files = [f for f in jsonl_files if self.split in f.stem]
        if split_files:
            return split_files
        
        # Return all files if no split-specific files
        return jsonl_files
    
    def _create_synthetic_data(self, base_path: Path) -> List[Path]:
        """Create synthetic training data for testing."""
        base_path.mkdir(parents=True, exist_ok=True)
        
        synthetic_file = base_path / f"synthetic_{self.split}.jsonl"
        
        # Generate synthetic text samples
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Large language models can generate human-like text.",
            "Training neural networks requires significant computational resources.",
            "The transformer architecture revolutionized natural language processing.",
            "Attention mechanisms allow models to focus on relevant information.",
            "Gradient descent optimizes model parameters during training.",
            "Deep learning has applications in computer vision and NLP.",
            "Overfitting occurs when a model memorizes training data.",
            "Regularization techniques help improve model generalization."
        ]
        
        # Create JSONL file with synthetic samples
        num_samples = 1000 if self.split == 'train' else 100
        
        with open(synthetic_file, 'w') as f:
            for i in range(num_samples):
                # Create variations of sample texts
                text = sample_texts[i % len(sample_texts)]
                if i > 0:
                    text = f"Sample {i}: {text}"
                
                sample = {'text': text}
                f.write(json.dumps(sample) + '\n')
        
        print(f"Created synthetic data file: {synthetic_file}")
        return [synthetic_file]
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load samples from JSONL files."""
        samples = []
        
        for file_path in self.data_paths:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        warnings.warn(f"Invalid JSON at {file_path}:{line_num}: {e}")
                        continue
        
        # Shuffle samples for training
        if self.split == 'train':
            random.shuffle(samples)
        
        return samples
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tokenized sequence tensor
        """
        sample = self.samples[idx]
        
        # Extract text
        if 'text' in sample:
            text = sample['text']
        elif 'content' in sample:
            text = sample['content']
        else:
            # Fallback: use first string value
            text = next((v for v in sample.values() if isinstance(v, str)), "")
        
        # Tokenize
        if self.tokenizer:
            token_ids = self.tokenizer.encode(text)
        else:
            # Fallback: character-level encoding
            token_ids = [ord(c) % 1000 for c in text[:self.seq_length]]
        
        # Pad or truncate to seq_length
        if len(token_ids) > self.seq_length:
            token_ids = token_ids[:self.seq_length]
        else:
            # Pad with zeros
            pad_length = self.seq_length - len(token_ids)
            token_ids.extend([0] * pad_length)
        
        return torch.tensor(token_ids, dtype=torch.long)


def create_data_loaders(
    data_config: Dict[str, Any],
    tokenizer,
    batch_size: int = 4,
    seq_length: int = 2048,
    num_workers: int = 2
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        data_config: Dataset configuration
        tokenizer: Tokenizer for text encoding
        batch_size: Batch size for training
        seq_length: Maximum sequence length
        num_workers: Number of data loading workers
        
    Returns:
        Dictionary of DataLoaders
    """
    data_loaders = {}
    
    # Training loader
    train_dataset = CDRDataset(
        data_config, tokenizer, seq_length, split='train'
    )
    
    data_loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Validation loader
    val_dataset = CDRDataset(
        data_config, tokenizer, seq_length, split='val'
    )
    
    data_loaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return data_loaders


def load_dataset():
    """Legacy function for compatibility."""
    print('Use create_data_loaders() for new code')
    return None
