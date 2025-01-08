import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Tuple, Dict

class GraphReasoningEngine(nn.Module):
    """Graph neural network engine for mathematical concept reasoning"""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                concat=True,
                dropout=0.1
            ) for _ in range(3)
        ])
        
        # Edge attention
        self.edge_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Global pooling projection
        self.global_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initial attention weights
        attention_scores = []
        
        # Process through GAT layers
        for gat_layer in self.gat_layers:
            x, attention = gat_layer(x, edge_index, return_attention_weights=True)
            attention_scores.append(attention)
            x = torch.relu(x)
        
        # Global graph representation
        global_graph = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))
        global_graph = self.global_projection(global_graph)
        
        return x, attention_scores