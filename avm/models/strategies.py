import torch
import torch.nn as nn
from typing import Tuple

class ProofStrategy(nn.Module):
    """Base class for all proof strategies"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def get_confidence(self, features: torch.Tensor) -> torch.Tensor:
        return self.confidence_net(features.mean(dim=1))

class DirectProofStrategy(ProofStrategy):
    def __init__(self, hidden_dim: int):
        super().__init__(hidden_dim)
        self.step_generator = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate direct proof steps
        steps = self.step_generator(features, features)
        confidence = self.get_confidence(features)
        return steps, confidence

class ContradictionStrategy(ProofStrategy):
    def __init__(self, hidden_dim: int):
        super().__init__(hidden_dim)
        self.negation = nn.Linear(hidden_dim, hidden_dim)
        self.step_generator = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Negate assumption and find contradiction
        negated = self.negation(features)
        steps = self.step_generator(negated, features)
        confidence = self.get_confidence(features)
        return steps, confidence

class InductionStrategy(ProofStrategy):
    def __init__(self, hidden_dim: int):
        super().__init__(hidden_dim)
        self.base_case = nn.Linear(hidden_dim, hidden_dim)
        self.inductive_step = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate base case and inductive step
        base = self.base_case(features)
        steps = self.inductive_step(base, features)
        confidence = self.get_confidence(features)
        return steps, confidence

class ConstructionStrategy(ProofStrategy):
    def __init__(self, hidden_dim: int):
        super().__init__(hidden_dim)
        self.constructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Construct object with desired properties
        steps = self.constructor(features)
        confidence = self.get_confidence(features)
        return steps, confidence

class ReductionStrategy(ProofStrategy):
    def __init__(self, hidden_dim: int):
        super().__init__(hidden_dim)
        self.reducer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reduce to simpler known problem
        steps = self.reducer(features, features)
        confidence = self.get_confidence(features)
        return steps, confidence