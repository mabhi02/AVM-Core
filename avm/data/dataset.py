import torch
from torch.utils.data import Dataset
import json
from typing import Dict, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ProofDataset(Dataset):
    def __init__(self, proofs_file: str, vocab_file: str, max_len: int = 512):
        """
        Args:
            proofs_file: JSON file containing proofs
            vocab_file: JSON file containing vocabulary
            max_len: Maximum sequence length
        """
        logger.info(f"Loading proofs from {proofs_file}")
        with open(proofs_file, 'r', encoding='utf-8') as f:
            self.proofs = json.load(f)
            
        logger.info(f"Loading vocabulary from {vocab_file}")
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            
        self.symbol2idx = {sym: idx for idx, sym in enumerate(self.vocab['symbols'])}
        self.idx2symbol = {idx: sym for sym, idx in self.symbol2idx.items()}
        self.max_len = max_len
        
        logger.info(f"Loaded {len(self.proofs)} proofs and vocabulary of size {len(self.vocab['symbols'])}")
        
    def __len__(self) -> int:
        return len(self.proofs)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        proof = self.proofs[idx]
        
        # Convert symbols to indices
        symbols = proof.get('symbols', [])
        symbol_indices = [self.symbol2idx.get(s, self.symbol2idx['<UNK>']) 
                         for s in symbols]
        symbol_tensor = torch.tensor(symbol_indices, dtype=torch.long)
        
        # Convert proof steps
        steps = proof.get('steps', [])
        step_tensors = []
        for step in steps:
            step_tensor = self.encode_step(step)
            step_tensors.append(step_tensor)
            
        if step_tensors:
            steps_tensor = torch.stack(step_tensors)
        else:
            # Create dummy tensor if no steps
            steps_tensor = torch.zeros((1, self.max_len), dtype=torch.long)
        
        # Create edge information
        relations = proof.get('relations', [])
        if relations:
            edge_index = []
            edge_type = []
            for rel in relations:
                edge_index.append([rel['from_idx'], rel['to_idx']])
                edge_type.append(rel.get('type_idx', 0))
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_type = torch.tensor(edge_type, dtype=torch.long)
        else:
            # Create dummy tensors if no relations
            edge_index = torch.zeros((2, 1), dtype=torch.long)
            edge_type = torch.zeros(1, dtype=torch.long)
            
        return {
            'symbols': symbol_tensor,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'steps': steps_tensor,
            'theorem': self.encode_step(proof['theorem']),
            'strategy_labels': torch.tensor(proof.get('strategy_labels', [0])),
            'difficulty': torch.tensor(proof.get('difficulty', 0.5), dtype=torch.float)
        }
    
    def encode_step(self, text: str) -> torch.Tensor:
        """Encode a proof step into tensor representation"""
        tokens = text.split()
        indices = [self.symbol2idx.get(token, self.symbol2idx['<UNK>']) 
                  for token in tokens]
        
        # Pad or truncate to max_len
        if len(indices) < self.max_len:
            indices += [self.symbol2idx['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
            
        return torch.tensor(indices, dtype=torch.long)