import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupScheduler(_LRScheduler):
    """Learning rate scheduler with warmup and cosine decay"""
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_steps: int,
                 max_steps: int,
                 min_lr: float = 1e-6):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer)
        
    def get_lr(self) -> List[float]:
        step = self.last_epoch
        
        # Warmup phase
        if step < self.warmup_steps:
            return [base_lr * (step / self.warmup_steps) 
                    for base_lr in self.base_lrs]
        
        # Cosine decay phase
        step = min(step, self.max_steps)
        cos_factor = 0.5 * (1 + math.cos(math.pi * step / self.max_steps))
        
        return [self.min_lr + (base_lr - self.min_lr) * cos_factor
                for base_lr in self.base_lrs]
