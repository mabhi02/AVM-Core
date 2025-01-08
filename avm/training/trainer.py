import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
import logging
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)

def setup_memory_efficient_training(model, config):
    """Setup memory efficient training configurations."""
    import torch
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Enable cudnn benchmarking for faster training
    torch.backends.cudnn.benchmark = True
    
    # Update model config for memory efficiency
    model_updates = {
        'max_len': 2048,  # Reduce from 5000 to 2048
        'hidden_dim': 256,  # Reduce if currently higher
        'num_heads': 8,  # Keep divisible by hidden_dim
        'batch_size': 8,  # Reduce from 32
        'gradient_accumulation_steps': 4  # Simulate larger batch size
    }
    
    # Update config
    if 'model' not in config:
        config['model'] = {}
    config['model'].update(model_updates)
    
    if 'training' not in config:
        config['training'] = {}
    config['training']['batch_size'] = model_updates['batch_size']
    config['training']['gradient_accumulation_steps'] = model_updates['gradient_accumulation_steps']
    
    logger.info("Memory efficient training settings applied:")
    logger.info(f"Max sequence length: {model_updates['max_len']}")
    logger.info(f"Batch size: {model_updates['batch_size']}")
    logger.info(f"Gradient accumulation steps: {model_updates['gradient_accumulation_steps']}")
    
    return config

class MemoryEfficientTrainer:
    """Memory efficient training wrapper."""
    def __init__(self, 
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer: AdamW,
                 scheduler: CosineAnnealingLR,
                 config: dict,
                 device: torch.device):
        """Initialize trainer with model and training components."""
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Training settings
        training_config = config['training']
        self.max_epochs = training_config['max_epochs']
        self.gradient_clip_val = training_config['gradient_clip_val']
        self.validation_interval = training_config['validation_interval']
        self.checkpoint_dir = training_config['checkpoint_dir']
        self.early_stopping_patience = training_config['early_stopping_patience']
        self.gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        
        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        
        # State tracking
        self.start_epoch = 1
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Print trainer setup info
        logger.info(f"Trainer initialized with {len(train_dataloader)} training batches")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Using device: {device}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        try:
            for epoch in range(self.start_epoch, self.max_epochs + 1):
                self.current_epoch = epoch
                
                # Training epoch
                train_loss = self.train_epoch(epoch)
                logger.info(f"Epoch {epoch} training loss: {train_loss:.4f}")
                
                # Validation
                if epoch % self.validation_interval == 0:
                    val_loss = self.validate()
                    logger.info(f"Epoch {epoch} validation loss: {val_loss:.4f}")
                    
                    # Check for improvement
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        self.save_checkpoint(epoch, val_loss)
                    else:
                        self.patience_counter += 1
                        
                    # Early stopping
                    if self.patience_counter >= self.early_stopping_patience:
                        logger.info("Early stopping triggered")
                        break
                
                # Update learning rate
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Learning rate: {current_lr:.6f}")
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_interrupt_checkpoint()
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def train_epoch(self, epoch: int) -> float:
        """Memory efficient training for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create progress bar
        pbar = tqdm(total=len(self.train_dataloader), 
                   desc=f"Epoch {epoch}",
                   dynamic_ncols=True)
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"GPU Memory at start: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            try:
                # Prepare batch with truncation
                batch = self._prepare_and_truncate_batch(batch)
                
                # Use automatic mixed precision
                with torch.cuda.amp.autocast():
                    # Forward pass
                    outputs = self.model(batch)
                    loss_dict = self.model.calculate_loss(batch)
                    loss = loss_dict['loss'] / self.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Unscale gradients and clip
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                # Update metrics
                current_loss = loss.item() * self.gradient_accumulation_steps
                total_loss += current_loss
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{current_loss:.4f}",
                    'gpu_mem': f"{torch.cuda.memory_allocated()/1024**2:.1f}MB"
                })
                pbar.update(1)
                
                # Memory cleanup
                del outputs, loss_dict, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}:")
                logger.error(str(e))
                raise e
        
        pbar.close()
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                batch = self._prepare_and_truncate_batch(batch)
                loss_dict = self.model.calculate_loss(batch)
                total_loss += loss_dict['loss'].item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _prepare_and_truncate_batch(self, batch):
        """Prepare and truncate batch to fit in memory."""
        max_len = self.config['model']['max_len']
        
        prepared_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if len(value.shape) >= 2 and value.shape[1] > max_len:
                    if len(value.shape) == 3:
                        value = value[:, :, :max_len]
                    else:
                        value = value[:, :max_len]
                
                if key in ['symbols', 'theorem', 'steps', 'strategy_labels', 'edge_index', 'edge_type']:
                    prepared_batch[key] = value.to(self.device, dtype=torch.long)
                else:
                    prepared_batch[key] = value.to(self.device, dtype=torch.float32)
            else:
                prepared_batch[key] = value
                
        return prepared_batch
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pt')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_interrupt_checkpoint(self):
        """Save checkpoint on interrupt."""
        checkpoint_path = os.path.join(self.checkpoint_dir, 'interrupt_checkpoint.pt')
        self.save_checkpoint(self.current_epoch, self.best_val_loss)