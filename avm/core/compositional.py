from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
from .graph_engine import GraphReasoningEngine

class CompositionalEngine(nn.Module):
    """Core compositional reasoning engine for AVM-CORE"""
    def __init__(self, 
                 hidden_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Graph reasoning component
        self.graph_engine = GraphReasoningEngine(
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        
        # Compositional transformer
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4*hidden_dim,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        
        # Component integration
        self.integration_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self,
                input_embeddings: torch.Tensor,
                graph_data: Dict[str, torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        # Process graph structure
        graph_embeddings, graph_attention = self.graph_engine(
            graph_data['nodes'],
            graph_data['edges'],
            graph_data['edge_attr']
        )
        
        # Process sequential data
        sequential_embeddings = self.transformer(
            input_embeddings,
            src_key_padding_mask=mask
        )
        
        # Integrate components
        combined = self.integration_layer(
            torch.cat([
                sequential_embeddings,
                graph_embeddings.unsqueeze(0).expand(sequential_embeddings.size(0), -1, -1)
            ], dim=-1)
        )
        
        return combined, {
            'graph_attention': graph_attention,
            'sequential_attention': self.transformer.layers[-1].self_attn
        }