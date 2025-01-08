import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional

class AdaptiveLROptimizer:
    """Custom optimizer with adaptive learning rate based on validation performance"""
    def __init__(self, 
                 model_params,
                 initial_lr: float = 1e-4,
                 warmup_steps: int = 1000,
                 min_lr: float = 1e-6):
        self.optimizer = torch.optim.AdamW(
            model_params,
            lr=initial_lr,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.01
        )
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.current_step = 0
        
    def step(self, validation_metrics: Optional[Dict] = None):
        """Take an optimization step with adaptive LR"""
        self.current_step += 1
        
        # Warmup phase
        if self.current_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.current_step) / self.warmup_steps)
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.min_lr + (pg['initial_lr'] - self.min_lr) * lr_scale
        
        # Adapt based on validation metrics
        elif validation_metrics is not None:
            self._adapt_lr(validation_metrics)
            
        self.optimizer.step()
        
    def _adapt_lr(self, metrics: Dict):
        """Adapt learning rate based on validation metrics"""
        if 'validation_loss' in metrics:
            # Simple adaptation strategy
            if metrics.get('validation_loss_decreased', False):
                scale_factor = 1.0
            else:
                scale_factor = 0.95
                
            for pg in self.optimizer.param_groups:
                pg['lr'] = max(self.min_lr, pg['lr'] * scale_factor)