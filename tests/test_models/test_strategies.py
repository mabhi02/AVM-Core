import pytest
import torch
from avm.models.strategies import DirectProofStrategy, ContradictionStrategy

class TestProofStrategies:
    @pytest.fixture
    def strategies(self):
        return {
            'direct': DirectProofStrategy(hidden_dim=256),
            'contradiction': ContradictionStrategy(hidden_dim=256)
        }
    
    def test_direct_strategy(self, strategies):
        batch_size = 4
        state = torch.randn(batch_size, 256)
        
        output, confidence = strategies['direct'](state)
        
        assert output.shape[-1] == 256
        assert confidence.shape == (batch_size, 1)
    
    def test_strategy_confidence(self, strategies):
        state = torch.randn(4, 256)
        
        for strategy in strategies.values():
            confidence = strategy.get_confidence(state)
            assert torch.all(confidence >= 0) and torch.all(confidence <= 1)