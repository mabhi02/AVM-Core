import argparse
import yaml
import torch
from pathlib import Path
from avm.models.proof_generator import AVMCore
from avm.training.trainer import ProofTrainer
from avm.data.dataset import ProofDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train AVM-CORE')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                      help='Path to training configuration file')
    parser.add_argument('--model-config', type=str, default='configs/model_config.yaml',
                      help='Path to model configuration file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configurations
    with open(args.config, 'r') as f:
        train_config = yaml.safe_load(f)
    
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
        
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize model
    model = AVMCore(**model_config['model'])
    
    # Initialize data
    train_dataset = ProofDataset(train_config['data']['train_path'])
    val_dataset = ProofDataset(train_config['data']['val_path'])
    
    # Initialize trainer
    trainer = ProofTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        **train_config['training']
    )
    
    # Train model
    trainer.train()

if __name__ == '__main__':
    main()