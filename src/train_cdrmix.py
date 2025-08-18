"""
CDRmix Training Script

Complete training pipeline for CDRmix models with MoE routing.
Supports all model scales (1B, 4B, 40B, 200B) with proper MoE loss integration.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
import time

# CDRmix imports
from cdrmix.cdrmix_model import CDRmixConfig, CDRmixModel
from data_loader import create_data_loaders
from utils.config import load_yaml_config, get_model_config_from_yaml, get_training_config
from utils.tokenizer_utils import get_tokenizer_for_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CDRmixTrainer:
    """
    Complete training pipeline for CDRmix models.
    
    Features:
    - Mixed precision training
    - Gradient accumulation
    - MoE auxiliary loss integration
    - Checkpointing and resumption
    - RWKV state management
    """
    
    def __init__(self, config_path: str, resume_from: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to training configuration YAML
            resume_from: Optional checkpoint path to resume from
        """
        # Load configuration
        self.yaml_config = load_yaml_config(config_path)
        self.model_config = get_model_config_from_yaml(self.yaml_config)
        self.training_config = get_training_config(self.yaml_config)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self._setup_model()
        self._setup_tokenizer()
        self._setup_data()
        self._setup_optimizer()
        self._setup_training_state()
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        logger.info(f"Trainer initialized for {self.model_config['model_scale']} model")
    
    def _setup_model(self):
        """Initialize CDRmix model."""
        logger.info("Setting up CDRmix model...")
        
        # Create model configuration
        cdrmix_config = CDRmixConfig(**self.model_config)
        
        # Create model
        self.model = CDRmixModel(cdrmix_config)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model created: {cdrmix_config.model_scale}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Vocabulary size: {cdrmix_config.vocab_size}")
        logger.info(f"Model dimension: {cdrmix_config.d_model}")
        logger.info(f"Number of layers: {cdrmix_config.n_layers}")
        logger.info(f"Number of experts: {cdrmix_config.num_experts}")
    
    def _setup_tokenizer(self):
        """Initialize tokenizer."""
        logger.info("Setting up tokenizer...")
        self.tokenizer = get_tokenizer_for_config(self.yaml_config)
        logger.info(f"Tokenizer loaded with vocab_size: {self.tokenizer.vocab_size}")
    
    def _setup_data(self):
        """Initialize data loaders."""
        logger.info("Setting up data loaders...")
        
        # Get data config
        data_config = self.yaml_config.get('dataset', '/tmp/cdrmix_data')
        
        # Create data loaders
        seq_length = self.training_config.get('seq_length', 2048)
        batch_size = self.training_config.get('batch_size', 4)
        
        self.data_loaders = create_data_loaders(
            data_config=data_config,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            seq_length=seq_length,
            num_workers=2
        )
        
        logger.info(f"Data loaders created:")
        logger.info(f"  Train samples: {len(self.data_loaders['train'].dataset)}")
        logger.info(f"  Val samples: {len(self.data_loaders['val'].dataset)}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Sequence length: {seq_length}")
    
    def _setup_optimizer(self):
        """Initialize optimizer and scheduler."""
        logger.info("Setting up optimizer...")
        
        # Optimizer parameters
        lr = self.training_config.get('learning_rate', 3e-4)
        weight_decay = self.training_config.get('weight_decay', 0.01)
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        warmup_steps = self.training_config.get('warmup_steps', 2000)
        max_steps = self.training_config.get('max_steps', 100000)
        
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / warmup_steps) * (max_steps - step) / max_steps
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        
        logger.info(f"Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
        logger.info(f"Scheduler: Linear warmup + cosine decay")
        logger.info(f"Mixed precision: {self.scaler is not None}")
    
    def _setup_training_state(self):
        """Initialize training state tracking."""
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Training configuration
        self.max_steps = self.training_config.get('max_steps', 100000)
        self.eval_interval = self.training_config.get('eval_interval', 1000)
        self.save_interval = self.training_config.get('save_interval', 5000)
        self.gradient_clip_val = self.training_config.get('gradient_clip_val', 1.0)
        self.gradient_accumulation_steps = self.training_config.get('gradient_accumulation_steps', 1)
        
        # MoE loss configuration
        self.use_moe_aux_loss = self.training_config.get('use_moe_aux_loss', True)
        self.aux_loss_weight = self.training_config.get('aux_loss_weight', 0.01)
        
        # Loss criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        logger.info(f"Training configuration:")
        logger.info(f"  Max steps: {self.max_steps}")
        logger.info(f"  Eval interval: {self.eval_interval}")
        logger.info(f"  Save interval: {self.save_interval}")
        logger.info(f"  Gradient clipping: {self.gradient_clip_val}")
        logger.info(f"  MoE aux loss: {self.use_moe_aux_loss}")
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Execute single training step.
        
        Args:
            batch: Input batch [batch_size, seq_len]
            
        Returns:
            Dictionary of losses
        """
        self.model.train()
        
        # Prepare inputs and targets
        inputs = batch[:, :-1].to(self.device)  # [batch_size, seq_len-1]
        targets = batch[:, 1:].to(self.device)  # [batch_size, seq_len-1]
        
        losses = {}
        
        # Forward pass with mixed precision
        if self.scaler:
            with autocast():
                # Get model outputs
                if self.use_moe_aux_loss:
                    logits, states, aux_losses = self.model(
                        inputs, return_aux_losses=True, return_states=False
                    )
                else:
                    logits = self.model(
                        inputs, return_aux_losses=False, return_states=False
                    )
                
                # Compute primary loss
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                
                ce_loss = self.criterion(logits, targets)
                losses['ce_loss'] = ce_loss.item()
                
                # Add MoE auxiliary losses
                total_loss = ce_loss
                if self.use_moe_aux_loss and 'aux_losses' in locals():
                    aux_loss = (
                        aux_losses['load_balance'] + 
                        aux_losses['z_loss'] + 
                        aux_losses['overflow']
                    )
                    total_loss = ce_loss + self.aux_loss_weight * aux_loss
                    losses['aux_loss'] = aux_loss.item()
                
                losses['total_loss'] = total_loss.item()
        else:
            # Forward pass without mixed precision
            if self.use_moe_aux_loss:
                logits, states, aux_losses = self.model(
                    inputs, return_aux_losses=True, return_states=False
                )
            else:
                logits = self.model(
                    inputs, return_aux_losses=False, return_states=False
                )
            
            # Compute losses
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            ce_loss = self.criterion(logits, targets)
            losses['ce_loss'] = ce_loss.item()
            
            total_loss = ce_loss
            if self.use_moe_aux_loss and 'aux_losses' in locals():
                aux_loss = (
                    aux_losses['load_balance'] + 
                    aux_losses['z_loss'] + 
                    aux_losses['overflow']
                )
                total_loss = ce_loss + self.aux_loss_weight * aux_loss
                losses['aux_loss'] = aux_loss.item()
            
            losses['total_loss'] = total_loss.item()
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        return losses
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        progress_bar = tqdm(
            self.data_loaders['train'], 
            desc=f"Epoch {self.epoch+1}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            losses = self.train_step(batch)
            epoch_losses.append(losses)
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.step += 1
                
                # Update progress bar
                avg_loss = losses['total_loss']
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                    'step': self.step
                })
                
                # Evaluation and checkpointing
                if self.step % self.eval_interval == 0:
                    self.evaluate()
                
                if self.step % self.save_interval == 0:
                    self.save_checkpoint()
                
                # Check if training is complete
                if self.step >= self.max_steps:
                    logger.info(f"Training completed at step {self.step}")
                    return False
        
        return True
    
    def evaluate(self):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.data_loaders['val'], desc="Evaluating", leave=False):
                inputs = batch[:, :-1].to(self.device)
                targets = batch[:, 1:].to(self.device)
                
                # Forward pass
                logits = self.model(inputs, return_aux_losses=False, return_states=False)
                
                # Compute loss
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info(f"Step {self.step} - Val Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(is_best=True)
        
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save paths
        save_dir = Path('checkpoints')
        save_dir.mkdir(exist_ok=True)
        
        checkpoint_path = save_dir / f"checkpoint_step_{self.step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at step {self.step}")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Resumed training from step {self.step}")
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.max_steps} steps")
        
        start_time = time.time()
        
        try:
            while self.step < self.max_steps:
                continue_training = self.train_epoch()
                self.epoch += 1
                
                if not continue_training:
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            # Save final checkpoint
            self.save_checkpoint()
            
            # Training summary
            elapsed_time = time.time() - start_time
            logger.info(f"Training completed!")
            logger.info(f"Final step: {self.step}")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            logger.info(f"Total training time: {elapsed_time/3600:.2f} hours")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CDRmix model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/cdrmix-core-1b.yaml",
        help="Path to training configuration"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CDRmixTrainer(args.config, resume_from=args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()