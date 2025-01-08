import torch
from typing import Dict, List
import numpy as np

def calculate_proof_metrics(predictions: torch.Tensor, 
                          targets: torch.Tensor,
                          pad_idx: int = 0) -> Dict[str, float]:
    """Calculate various metrics for proof generation"""
    # Mask padded positions
    mask = (targets != pad_idx)
    
    # Accuracy
    correct = (predictions == targets) & mask
    accuracy = correct.sum().float() / mask.sum()
    
    # Per-step accuracy
    step_correct = correct.view(-1, targets.size(-1)).float().mean(dim=-1)
    step_accuracy = step_correct.mean()
    
    return {
        'token_accuracy': accuracy.item(),
        'step_accuracy': step_accuracy.item(),
    }

def calculate_strategy_metrics(predicted_weights: torch.Tensor,
                             actual_strategies: torch.Tensor) -> Dict[str, float]:
    """Calculate metrics for strategy selection"""
    predicted = predicted_weights.argmax(dim=-1)
    accuracy = (predicted == actual_strategies).float().mean()
    
    return {
        'strategy_accuracy': accuracy.item(),
        'strategy_diversity': len(predicted.unique())
    }