"""
Tokenizer utilities for CDRmix training.

Handles BPE tokenizer loading and text processing for training pipeline.
"""

import os
import json
from typing import Dict, List, Optional, Union
from pathlib import Path
import warnings


class CDRTokenizer:
    """
    Simple BPE tokenizer wrapper for CDRmix training.
    
    Provides a unified interface for tokenization compatible with
    the training pipeline expectations.
    """
    
    def __init__(self, vocab_path: str, merges_path: Optional[str] = None):
        """
        Initialize tokenizer from vocabulary files.
        
        Args:
            vocab_path: Path to vocabulary JSON file
            merges_path: Optional path to BPE merges file
        """
        self.vocab_path = Path(vocab_path)
        self.merges_path = Path(merges_path) if merges_path else None
        
        # Load vocabulary
        self._load_vocab()
        
        # Set special tokens
        self.pad_token_id = self.vocab.get('<pad>', 0)
        self.eos_token_id = self.vocab.get('<eos>', 1)
        self.bos_token_id = self.vocab.get('<bos>', 2)
        self.unk_token_id = self.vocab.get('<unk>', 3)
        
        self.vocab_size = len(self.vocab)
    
    def _load_vocab(self):
        """Load vocabulary from file."""
        if not self.vocab_path.exists():
            # Create minimal vocabulary for testing
            warnings.warn(f"Vocab file {self.vocab_path} not found. Creating minimal vocab for testing.")
            self._create_minimal_vocab()
            return
        
        try:
            with open(self.vocab_path, 'r') as f:
                self.vocab = json.load(f)
        except json.JSONDecodeError:
            warnings.warn(f"Invalid JSON in {self.vocab_path}. Creating minimal vocab for testing.")
            self._create_minimal_vocab()
    
    def _create_minimal_vocab(self):
        """Create minimal vocabulary for testing purposes."""
        # Create basic vocabulary with common tokens
        special_tokens = ['<pad>', '<eos>', '<bos>', '<unk>']
        
        # Add basic ASCII characters and common words
        chars = [chr(i) for i in range(32, 127)]  # Printable ASCII
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        
        vocab_list = special_tokens + chars + common_words
        
        # Add padding to reach reasonable vocab size
        while len(vocab_list) < 1000:
            vocab_list.append(f'<extra_token_{len(vocab_list)}>')
        
        self.vocab = {token: idx for idx, token in enumerate(vocab_list)}
        self.vocab_size = len(self.vocab)
        
        # Save minimal vocab for future use
        self.vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.vocab_path, 'w') as f:
            json.dump(self.vocab, f, indent=2)
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        # Simple character-level tokenization for now
        # In production, this would use proper BPE
        tokens = []
        
        for char in text:
            token_id = self.vocab.get(char, self.unk_token_id)
            tokens.append(token_id)
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        # Reverse vocabulary mapping
        if not hasattr(self, '_id_to_token'):
            self._id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        tokens = []
        for token_id in token_ids:
            if token_id == self.pad_token_id:
                continue  # Skip padding tokens
            token = self._id_to_token.get(token_id, '<unk>')
            tokens.append(token)
        
        return ''.join(tokens)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode batch of texts."""
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_ids_batch: List[List[int]]) -> List[str]:
        """Decode batch of token ID sequences."""
        return [self.decode(token_ids) for token_ids in token_ids_batch]


def load_tokenizer(tokenizer_path: str) -> CDRTokenizer:
    """
    Load tokenizer from path.
    
    Args:
        tokenizer_path: Path to tokenizer directory or vocab file
        
    Returns:
        CDRTokenizer instance
    """
    tokenizer_path = Path(tokenizer_path)
    
    if tokenizer_path.is_dir():
        # Look for vocab.json in directory
        vocab_path = tokenizer_path / 'vocab.json'
        merges_path = tokenizer_path / 'merges.txt'
        
        if not merges_path.exists():
            merges_path = None
    else:
        # Assume direct path to vocab file
        vocab_path = tokenizer_path
        merges_path = None
    
    return CDRTokenizer(str(vocab_path), str(merges_path) if merges_path else None)


def create_tokenizer_from_scratch(
    vocab_size: int = 50272,
    save_path: str = "/tmp/cdrmix_tokenizer"
) -> CDRTokenizer:
    """
    Create a minimal tokenizer for testing purposes.
    
    Args:
        vocab_size: Target vocabulary size
        save_path: Where to save the tokenizer
        
    Returns:
        CDRTokenizer instance
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create vocabulary
    special_tokens = ['<pad>', '<eos>', '<bos>', '<unk>', '<mask>']
    
    # Add printable ASCII
    ascii_chars = [chr(i) for i in range(32, 127)]
    
    # Add common subwords (simplified)
    common_subwords = []
    for i in range(256):  # Byte-level tokens
        common_subwords.append(f'<byte_{i}>')
    
    # Combine all tokens
    all_tokens = special_tokens + ascii_chars + common_subwords
    
    # Pad to target vocab size
    while len(all_tokens) < vocab_size:
        all_tokens.append(f'<extra_token_{len(all_tokens)}>')
    
    # Take only up to vocab_size
    all_tokens = all_tokens[:vocab_size]
    
    # Create vocab mapping
    vocab = {token: idx for idx, token in enumerate(all_tokens)}
    
    # Save vocabulary
    vocab_path = save_path / 'vocab.json'
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    # Create tokenizer
    tokenizer = CDRTokenizer(str(vocab_path))
    
    print(f"Created tokenizer with {len(vocab)} tokens at {save_path}")
    return tokenizer


def get_tokenizer_for_config(config: Dict) -> CDRTokenizer:
    """
    Get appropriate tokenizer for training configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        CDRTokenizer instance
    """
    if 'tokenizer_path' in config:
        tokenizer_path = config['tokenizer_path']
        if Path(tokenizer_path).exists():
            return load_tokenizer(tokenizer_path)
    
    # Fallback: create minimal tokenizer
    vocab_size = config.get('model', {}).get('vocab_size', 50272)
    print(f"Creating minimal tokenizer with vocab_size={vocab_size}")
    
    return create_tokenizer_from_scratch(vocab_size=vocab_size)