import pytest
import torch
from avm.core.compositional import CompositionalEngine

class TestCompositionalEngine:
    @pytest.fixture
    def engine(self):
        return CompositionalEngine(
            hidden_dim=256,
            num_heads=8
        )
    
    def test_forward_pass(self, engine):
        batch_size = 4
        seq_len = 10
        
        # Create dummy input data
        input_embeddings = torch.randn(batch_size, seq_len, 256)
        graph_data = {
            'nodes': torch.randn(batch_size, 20, 256),
            'edges': torch.randint(0, 20, (batch_size, 30, 2)),
            'edge_attr': torch.randn(batch_size, 30, 32)
        }
        
        # Forward pass
        output, attention_maps = engine(input_embeddings, graph_data)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, 256)
        assert 'graph_attention' in attention_maps
        assert 'sequential_attention' in attention_maps