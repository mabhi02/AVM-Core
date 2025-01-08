import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class ValidationEngine(nn.Module):
    """Validation engine for proof checking"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Step validation
        self.step_validator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Logical consistency checker
        self.consistency_checker = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4
            ),
            num_layers=3
        )
        
        # Global validation scorer
        self.global_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                proof_steps: torch.Tensor,
                proof_graph: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Validate individual steps
        step_scores = []
        for step in proof_steps:
            score = self.step_validator(
                torch.cat([step, proof_graph], dim=-1)
            )
            step_scores.append(score)
        step_scores = torch.stack(step_scores)
        
        # Check logical consistency
        consistency_features = self.consistency_checker(proof_steps)
        
        # Global validation score
        global_score = self.global_scorer(
            consistency_features.mean(dim=0)
        )
        
        return {
            'step_scores': step_scores,
            'global_score': global_score,
            'consistency_features': consistency_features
        }