import argparse
import yaml
import torch
from pathlib import Path
import logging
import os
from typing import Dict, Optional, Tuple, Union

from avm.models.proof_generator import AVMCore
from avm.data.dataset import ProofDataset
from avm.utils.tokenizer import ProofTokenizer
from avm.utils.collate import collate_proofs
from avm.training.trainer import MemoryEfficientTrainer, setup_memory_efficient_training

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('training.log')  # Also save to file
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AVM-CORE: Mathematical Proof Generation')
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
    
    # Add memory optimization arguments
    parser.add_argument('--batch-size', type=int,
                       help='Override batch size in config')
    parser.add_argument('--gradient-accumulation-steps', type=int,
                       help='Number of gradient accumulation steps')
    parser.add_argument('--chunk-size', type=int, default=512,
                       help='Size of chunks for memory efficient processing')
    parser.add_argument('--max-len', type=int,
                       help='Override maximum sequence length')
    parser.add_argument('--num-workers', type=int,
                       help='Number of dataloader workers')
    parser.add_argument('--pin-memory', action='store_true',
                       help='Use pinned memory for dataloaders')
    
    return parser.parse_args()

def update_configs(model_config: dict, train_config: dict, args) -> tuple:
    """Update both configurations with command line arguments."""
    # Update training config
    if args.batch_size is not None:
        train_config['training']['batch_size'] = args.batch_size
        logger.info(f"Overriding batch size to: {args.batch_size}")
        
    if args.gradient_accumulation_steps is not None:
        if 'training' not in train_config:
            train_config['training'] = {}
        train_config['training']['gradient_accumulation_steps'] = args.gradient_accumulation_steps
        logger.info(f"Setting gradient accumulation steps to: {args.gradient_accumulation_steps}")
        
    if args.num_workers is not None:
        if 'training' not in train_config:
            train_config['training'] = {}
        train_config['training']['num_workers'] = args.num_workers
        logger.info(f"Setting number of workers to: {args.num_workers}")
        
    if args.pin_memory:
        if 'training' not in train_config:
            train_config['training'] = {}
        train_config['training']['pin_memory'] = True
        logger.info("Enabling pinned memory")

    # Update model config
    if args.max_len is not None:
        if 'model' not in model_config:
            model_config['model'] = {}
        model_config['model']['max_len'] = args.max_len
        logger.info(f"Overriding max_len to: {args.max_len}")
        
    if args.chunk_size is not None:
        if 'model' not in model_config:
            model_config['model'] = {}
        if 'memory_efficient' not in model_config['model']:
            model_config['model']['memory_efficient'] = {}
        model_config['model']['memory_efficient']['max_chunk_size'] = args.chunk_size
        logger.info(f"Setting chunk size to: {args.chunk_size}")
        
    return model_config, train_config

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

def setup_environment(args):
    """Setup necessary directories and configurations."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load configurations
    model_config = load_config(args.config)
    train_config = load_config(args.train_config) if args.mode == 'train' else None
    
    return model_config, train_config, output_dir

def train(model: AVMCore, train_config: dict, train_dataset: ProofDataset, 
          val_dataset: ProofDataset, device: torch.device) -> MemoryEfficientTrainer:
    """Setup and initialize training."""
    
    # Apply memory optimizations
    train_config = setup_memory_efficient_training(model, train_config)
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_proofs
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_proofs
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['training']['learning_rate'],
        weight_decay=train_config['training']['optimizer']['weight_decay'],
        betas=(
            train_config['training']['optimizer']['beta1'],
            train_config['training']['optimizer']['beta2']
        )
    )
    
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config['training']['max_epochs'],
        eta_min=train_config['training']['scheduler']['min_lr']
    )
    
    # Initialize trainer
    trainer = MemoryEfficientTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=train_config,
        device=device
    )
    
    # Train
    trainer.train()
    
    return trainer

def evaluate(model: AVMCore, val_dataset: ProofDataset, device: torch.device) -> float:
    """Evaluation pipeline."""
    model.eval()
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8,  # Smaller batch size for evaluation
        shuffle=False,
        collate_fn=collate_proofs
    )
    
    total_loss = 0
    batch_count = 0
    
    progress_bar = tqdm(val_dataloader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in progress_bar:
            try:
                # Prepare batch
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                loss_dict = model.calculate_loss(batch)
                loss = loss_dict['loss'].item()
                total_loss += loss
                batch_count += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f"{loss:.4f}"})
                
            except Exception as e:
                logger.error(f"Error in evaluation batch: {str(e)}")
                raise
    
    progress_bar.close()            
    return total_loss / batch_count

def generate_proof(model: AVMCore, theorem: str, tokenizer: ProofTokenizer, device: torch.device) -> Dict:
    """Generate proof for a given theorem."""
    model.eval()
    with torch.no_grad():
        try:
            # Tokenize theorem
            theorem_tokens = tokenizer.encode(theorem).unsqueeze(0).to(device)
            
            # Generate proof
            outputs = model.generate_proof(theorem_tokens)
            
            # Decode and format results
            result = {
                'proof': tokenizer.decode(outputs['proof_tokens'][0]) if isinstance(outputs['proof_tokens'], torch.Tensor) else outputs['proof_tokens'],
                'confidence_scores': outputs.get('confidence_scores', None),
                'strategy_weights': outputs.get('strategy_weights', None),
                'intermediate_steps': outputs.get('intermediate_steps', [])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating proof: {str(e)}")
            raise

def save_output(output_path: Path, content: Dict):
    """Save outputs in a readable format."""
    with open(output_path, 'w') as f:
        if 'proof' in content:
            f.write(f"Proof:\n{content['proof']}\n\n")
            
        if 'strategy_analysis' in content:
            f.write("Strategy Analysis:\n")
            for strategy, details in content['strategy_analysis'].items():
                f.write(f"{strategy}: {details}\n")
            f.write("\n")
            
        if 'intermediate_steps' in content:
            f.write("Intermediate Steps:\n")
            for i, step in enumerate(content['intermediate_steps'], 1):
                f.write(f"Step {i}:\n{step}\n")
            f.write("\n")
            
        if 'confidence_scores' in content:
            f.write("Confidence Scores:\n")
            for i, score in enumerate(content['confidence_scores']):
                f.write(f"Step {i+1}: {score:.4f}\n")

def prepare_model_config(config: dict) -> dict:
    """Prepare model configuration."""
    model_params = {
        'hidden_dim': config['model']['hidden_dim'],
        'num_heads': config['model']['num_heads'],
        'num_layers': config['model']['num_layers'],
        'dropout': config['model']['dropout'],
        'vocab_size': config['model'].get('vocab_size', 50000),
        'max_len': config['model'].get('max_len', 4096),
        'beam_size': config['model'].get('beam_size', 5),
        'num_strategies': config['model'].get('num_strategies', 5),
    }
    
    if 'vocab_path' in config['model']:
        model_params['vocab_path'] = config['model']['vocab_path']
    
    return model_params

def setup_cuda_memory():
    """Setup CUDA memory management."""
    if torch.cuda.is_available():
        try:
            # Empty cache at start
            torch.cuda.empty_cache()
            
            # Basic CUDA initialization
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Log GPU info
            logger.info(f"GPU initialized: {torch.cuda.get_device_name(0)}")
            logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
            
        except Exception as e:
            logger.warning(f"CUDA setup warning: {str(e)}")
            logger.warning("Continuing with default CUDA settings")

def main():
    """Main execution function."""
    args = parse_args()
    model_config, train_config, output_dir = setup_environment(args)
    
    # Update both configs with command line arguments
    model_config, train_config = update_configs(model_config, train_config, args)
    
    try:
        # Setup CUDA before device initialization
        setup_cuda_memory()
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create model with updated config
        model_params = prepare_model_config(model_config)
        model = AVMCore(**model_params)
        model = model.to(device)
        
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        if args.mode == 'train':
            if not train_config:
                raise ValueError("Training configuration required for train mode")
                
            # Load datasets
            logger.info("Loading datasets...")
            train_dataset = ProofDataset(
                proofs_file=train_config['data']['train_path'],
                vocab_file=train_config['data']['vocab_path']
            )
            val_dataset = ProofDataset(
                proofs_file=train_config['data']['val_path'],
                vocab_file=train_config['data']['vocab_path']
            )
            
            # Train model and get trainer
            trainer = train(model, train_config, train_dataset, val_dataset, device)
            
            # Log training summary
            logger.info("\nTraining Summary:")
            logger.info(f"Total epochs: {trainer.current_epoch}")
            logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
            logger.info(f"Final learning rate: {trainer.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save final model with complete state
            final_path = os.path.join(train_config['training']['checkpoint_dir'], 'final_model.pt')
            torch.save({
                'epoch': trainer.current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'best_val_loss': trainer.best_val_loss,
                'model_config': model_config,
                'train_config': train_config
            }, final_path)
            logger.info(f"Final model and training state saved to {final_path}")
            
            if torch.cuda.is_available():
                logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**2:.1f}MB")
            
        elif args.mode == 'eval':
            if not train_config:
                raise ValueError("Training configuration required for eval mode")
                
            val_dataset = ProofDataset(
                proofs_file=train_config['data']['val_path'],
                vocab_file=train_config['data']['vocab_path']
            )
            eval_loss = evaluate(model, val_dataset, device)
            logger.info(f"Evaluation loss: {eval_loss:.4f}")
            
            # Save evaluation results
            eval_path = output_dir / "evaluation_results.txt"
            with open(eval_path, 'w') as f:
                f.write(f"Evaluation Loss: {eval_loss:.4f}\n")
            
            logger.info(f"Evaluation results saved to {eval_path}")
            
        elif args.mode == 'generate':
            if not args.theorem:
                raise ValueError("Theorem required for generate mode")
                
            if 'vocab_path' not in model_params:
                raise ValueError("Vocab path required for generation mode")
                
            tokenizer = ProofTokenizer(model_params['vocab_path'])
            
            logger.info(f"Generating proof for theorem: {args.theorem}")
            proof_result = generate_proof(model, args.theorem, tokenizer, device)
            
            # Save results with detailed information
            output_file = output_dir / "generated_proof.txt"
            with open(output_file, 'w') as f:
                f.write(f"Theorem: {args.theorem}\n\n")
                f.write(f"Generated Proof:\n{proof_result['proof']}\n\n")
                if proof_result.get('confidence_scores'):
                    f.write(f"Confidence Scores:\n")
                    for step, score in enumerate(proof_result['confidence_scores']):
                        f.write(f"Step {step + 1}: {score:.4f}\n")
                
                if proof_result.get('strategy_weights') is not None:
                    f.write("\nStrategy Distribution:\n")
                    strategies = ["direct", "contradiction", "induction", "construction"]
                    weights = proof_result['strategy_weights'].cpu().numpy()
                    for strategy, weight in zip(strategies, weights[0]):
                        f.write(f"{strategy}: {weight:.4f}\n")
                
                if proof_result.get('intermediate_steps'):
                    f.write("\nIntermediate Steps:\n")
                    for i, step in enumerate(proof_result['intermediate_steps'], 1):
                        f.write(f"Step {i}:\n{step}\n")
                    
            logger.info(f"Generated proof saved to {output_file}")
            
        elif args.test:
            logger.info("Running test forward pass...")
            # Create test batch
            batch = {
                'theorem': torch.randint(0, model_params['vocab_size'], (2, 128)).to(device),
                'steps': torch.randint(0, model_params['vocab_size'], (2, 10, 128)).to(device),
                'strategy_labels': torch.randint(0, 5, (2, 10)).to(device),
                'symbols': torch.randint(0, model_params['vocab_size'], (2, 128)).to(device),
                'edge_index': torch.randint(0, 2, (2, 10)).to(device),
                'edge_type': torch.zeros(10).to(device),
                'difficulty': torch.rand(2).to(device)
            }
            
            try:
                outputs = model(batch)
                logger.info("Test forward pass successful!")
                logger.info("\nOutput shapes:")
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        logger.info(f"{k}: {v.shape}")
                
            except Exception as e:
                logger.error(f"Error during test forward pass: {str(e)}")
                logger.error("\nInput batch shapes:")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        logger.error(f"{k}: shape={v.shape}, dtype={v.dtype}")
                raise
            
            logger.info("Test completed successfully")
            
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        if torch.cuda.is_available():
            logger.error(f"GPU Memory at error: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        raise

if __name__ == '__main__':
    main()