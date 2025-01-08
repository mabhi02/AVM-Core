import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from pathlib import Path
import os
import logging
from typing import Dict, Optional, Union, Tuple

from avm.models.proof_generator import AVMCore 
from avm.data.dataset import ProofDataset
from avm.utils.tokenizer import ProofTokenizer
from avm.utils.collate import collate_proofs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProofTrainer:
    def __init__(self,
                 model: AVMCore,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer: AdamW,
                 scheduler: CosineAnnealingLR,
                 config: dict,
                 device: torch.device):
        """Initialize trainer with model and training components."""
        # Core components
        self.model = model.to(device)
        self.model = self.model.type(torch.float32)  # Ensure float32
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Training settings from config
        training_config = config['training']
        self.max_epochs = training_config['max_epochs']
        self.gradient_clip_val = training_config['gradient_clip_val']
        self.validation_interval = training_config['validation_interval']
        self.checkpoint_dir = training_config['checkpoint_dir']
        self.early_stopping_patience = training_config['early_stopping_patience']
        
        # State tracking
        self.start_epoch = 1
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare batch with proper types and device placement."""
        prepared_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device, dtype=torch.float32)
            elif isinstance(value, (list, tuple)):
                prepared_batch[key] = [
                    v.to(self.device, dtype=torch.float32) if isinstance(v, torch.Tensor) else v 
                    for v in value
                ]
            elif isinstance(value, dict):
                prepared_batch[key] = {
                    k: v.to(self.device, dtype=torch.float32) if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                prepared_batch[key] = value
        
        return prepared_batch

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Number of training batches: {len(self.train_dataloader)}")
        
        try:
            for epoch in range(self.start_epoch, self.max_epochs + 1):
                self.current_epoch = epoch
                logger.info(f"Starting epoch {epoch}/{self.max_epochs}")
                
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
                        # Save best model
                        checkpoint_path = os.path.join(self.checkpoint_dir, f'best_model_epoch_{epoch}.pt')
                        self.save_checkpoint(epoch, val_loss, checkpoint_path)
                        logger.info(f"Saved best model checkpoint at epoch {epoch}")
                    else:
                        self.patience_counter += 1
                        logger.info(f"No improvement for {self.patience_counter} epochs")

                    # Early stopping
                    if self.patience_counter >= self.early_stopping_patience:
                        logger.info("Early stopping triggered.")
                        break

                # Update learning rate
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Current learning rate: {current_lr:.6f}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_interrupt_checkpoint()
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        processed_batches = 0
        
        progress_bar = tqdm(
            total=len(self.train_dataloader),
            desc=f"Epoch {epoch}/{self.max_epochs}",
            leave=True,
            ncols=100,
            position=0
        )

        try:
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Prepare batch
                batch = self._prepare_batch(batch)
                
                # Forward and backward pass
                self.optimizer.zero_grad()
                loss_dict = self.model.calculate_loss(batch)
                loss = loss_dict['loss']
                
                # Backward pass with gradient clipping
                loss.backward()
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.gradient_clip_val
                    )
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                processed_batches += 1
                avg_loss = total_loss / processed_batches
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                progress_bar.update()

        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            self._log_batch_error(batch)
            raise
        finally:
            progress_bar.close()
        
        return total_loss / len(self.train_dataloader)

    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        processed_batches = 0
        
        progress_bar = tqdm(
            total=len(self.val_dataloader),
            desc=f"Validation Epoch {self.current_epoch}/{self.max_epochs}",
            leave=True,
            ncols=100,
            position=0
        )

        try:
            with torch.no_grad():
                for batch in self.val_dataloader:
                    # Prepare and validate batch
                    batch = self._prepare_batch(batch)
                    loss_dict = self.model.calculate_loss(batch)
                    loss = loss_dict['loss']
                    
                    # Update metrics
                    total_loss += loss.item()
                    processed_batches += 1
                    avg_loss = total_loss / processed_batches
                    
                    # Update progress bar
                    progress_bar.set_postfix({'val_loss': f"{avg_loss:.4f}"})
                    progress_bar.update()

        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            raise
        finally:
            progress_bar.close()
        
        return total_loss / len(self.val_dataloader)

    def save_checkpoint(self, epoch: int, val_loss: float, path: Union[str, Path]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def _save_interrupt_checkpoint(self):
        """Save checkpoint on interrupt."""
        interrupt_path = Path(self.checkpoint_dir) / "interrupt_checkpoint.pt"
        self.save_checkpoint(
            epoch=self.current_epoch,
            val_loss=self.best_val_loss,
            path=interrupt_path
        )

    def _log_batch_error(self, batch: Dict[str, torch.Tensor]):
        """Log batch information on error."""
        logger.error("Batch contents:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                logger.error(f"{k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, tuple):
                logger.error(f"{k}: (shape: {v[0].shape}, {v[1].shape}), "
                           f"dtype: ({v[0].dtype}, {v[1].dtype})")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train AVM-CORE')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'generate'],
                       help='Operation mode')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--train-config', type=str, default='configs/training_config.yaml',
                       help='Path to training configuration')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--theorem', type=str, default=None,
                       help='Theorem to prove (for generate mode)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory for outputs')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode')
    return parser.parse_args()

def validate_config(config: dict) -> dict:
    """Validate and convert config values to proper types."""
    if 'training' in config:
        train_config = config['training']
        train_config['learning_rate'] = float(train_config['learning_rate'])
        
        if 'optimizer' in train_config:
            opt_config = train_config['optimizer']
            opt_config['weight_decay'] = float(opt_config['weight_decay'])
            opt_config['beta1'] = float(opt_config['beta1'])
            opt_config['beta2'] = float(opt_config['beta2'])
            
        if 'scheduler' in train_config:
            sched_config = train_config['scheduler']
            sched_config['min_lr'] = float(sched_config['min_lr'])
            
    return config

def load_config(config_path: str) -> dict:
    """Load and validate configuration."""
    logger.info(f"Loading config from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return validate_config(config)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise

def setup_model_and_training(config: dict, train_config: dict, device: torch.device) -> Tuple[AVMCore, DataLoader, DataLoader, AdamW, CosineAnnealingLR]:
    """Setup model, datasets, and training components."""
    # Initialize model
    model_params = {
        'hidden_dim': config['model']['hidden_dim'],
        'num_heads': config['model']['num_heads'],
        'num_layers': config['model']['num_layers'],
        'dropout': config['model']['dropout'],
        'vocab_size': config['model'].get('vocab_size', 50000),
        'max_len': config['model'].get('max_len', 512),
        'beam_size': config['model'].get('beam_size', 5),
    }
    
    if train_config['data'].get('vocab_path'):
        model_params['vocab_path'] = train_config['data']['vocab_path']
    
    model = AVMCore(**model_params).to(device)
    
    # Initialize datasets
    train_dataset = ProofDataset(
        proofs_file=train_config['data']['train_path'],
        vocab_file=train_config['data']['vocab_path']
    )
    val_dataset = ProofDataset(
        proofs_file=train_config['data']['val_path'],
        vocab_file=train_config['data']['vocab_path']
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_proofs
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_proofs
    )
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_config['training']['learning_rate'],
        weight_decay=train_config['training']['optimizer']['weight_decay'],
        betas=(
            train_config['training']['optimizer']['beta1'],
            train_config['training']['optimizer']['beta2']
        )
    )
    
    # Initialize scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_config['training']['max_epochs'],
        eta_min=train_config['training']['scheduler']['min_lr']
    )
    
    return model, train_dataloader, val_dataloader, optimizer, scheduler

def main():
    """Main training script."""
    args = parse_args()
    
    try:
        # Load configurations
        model_config = load_config(args.config)
        train_config = load_config(args.train_config)
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Setup directories
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup model and training components
        model, train_dataloader, val_dataloader, optimizer, scheduler = setup_model_and_training(
            model_config, train_config, device
        )
        
# Load checkpoint if specified
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resuming from epoch {start_epoch}")

        # Initialize trainer
        trainer = ProofTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=train_config,
            device=device
        )

        # Handle different modes
        if args.mode == 'train':
            logger.info("Starting training...")
            try:
                trainer.train()
                
            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                # Save interrupt checkpoint
                interrupt_path = output_dir / "interrupt_checkpoint.pt"
                trainer.save_checkpoint(
                    epoch=trainer.current_epoch,
                    val_loss=trainer.best_val_loss,
                    path=interrupt_path
                )
                
            except Exception as e:
                logger.error(f"Error during training: {str(e)}")
                raise
                
            finally:
                # Save final model state
                final_path = output_dir / "final_model.pt"
                trainer.save_checkpoint(
                    epoch=train_config['training']['max_epochs'],
                    val_loss=trainer.best_val_loss,
                    path=final_path
                )
                
                # Generate evaluation examples if specified
                if train_config['evaluation']['generate_examples']:
                    logger.info("Generating evaluation examples...")
                    eval_path = output_dir / "evaluation_examples_final.txt"
                    trainer.generate_examples(
                        num_examples=train_config['evaluation']['num_examples'],
                        save_path=eval_path
                    )

        elif args.mode == 'eval':
            logger.info("Running evaluation...")
            val_loss = trainer.validate()
            logger.info(f"Validation loss: {val_loss:.4f}")
            
        elif args.mode == 'generate':
            if not args.theorem:
                raise ValueError("Theorem required for generate mode")
                
            logger.info("Generating proof...")
            tokenizer = ProofTokenizer(train_config['data']['vocab_path'])
            proof_result = model.generate_proof(
                theorem_tokens=tokenizer.encode(args.theorem).unsqueeze(0).to(device),
                max_steps=train_config['data']['max_proof_steps']
            )
            
            # Save results
            output_file = output_dir / "generated_proof.txt"
            with open(output_file, 'w') as f:
                f.write(f"Theorem: {args.theorem}\n\n")
                f.write(f"Generated Proof:\n{proof_result['proof']}\n\n")
                f.write(f"Confidence: {proof_result['confidence']:.4f}\n")
                f.write("Strategy Analysis:\n")
                for k, v in proof_result['strategy_analysis'].items():
                    f.write(f"{k}: {v}\n")
                    
            logger.info(f"Generated proof saved to {output_file}")
            
        elif args.test:
            logger.info("Running test forward pass...")
            batch = {
                'theorem': torch.randint(0, model_config['model'].get('vocab_size', 50000), (2, 128)).float().to(device),
                'steps': torch.randint(0, model_config['model'].get('vocab_size', 50000), (2, 10, 128)).float().to(device),
                'hypothesis': torch.randint(0, model_config['model'].get('vocab_size', 50000), (2, 128)).float().to(device),
                'proof_steps': torch.randint(0, model_config['model'].get('vocab_size', 50000), (2, 10, 128)).float().to(device),
                'symbol_pairs': (
                    torch.randint(0, model_config['model'].get('vocab_size', 50000), (2, 128)).float().to(device),
                    torch.randint(0, model_config['model'].get('vocab_size', 50000), (2, 128)).float().to(device)
                ),
                'relation_types': torch.zeros((2, 128)).float().to(device)
            }
            
            try:
                outputs = model(batch)
                logger.info("Test forward pass successful!")
                logger.info("Output shapes:")
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        logger.info(f"{k}: {v.shape}")
            except Exception as e:
                logger.error(f"Error during test forward pass: {str(e)}")
                logger.info("Input batch shapes:")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        logger.info(f"{k}: {v.shape}, dtype={v.dtype}")
                    elif isinstance(v, tuple):
                        logger.info(f"{k}: (shape: {v[0].shape}, {v[1].shape}), "
                                  f"dtype: ({v[0].dtype}, {v[1].dtype})")
                raise

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()