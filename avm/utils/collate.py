import torch
import json
import logging
from typing import List, Dict, Union
from ..utils.tokenizer import ProofTokenizer

logger = logging.getLogger(__name__)

class ProofDataset(torch.utils.data.Dataset):
    """Dataset for proof generation with target handling."""
    def __init__(self, proofs_file: str, vocab_file: str):
        logger.info(f"Loading proofs from {proofs_file}")
        with open(proofs_file, 'r') as f:
            self.proofs = json.load(f)
            
        logger.info(f"Loading vocabulary from {vocab_file}")
        self.tokenizer = ProofTokenizer(vocab_file)
        logger.info(f"Loaded {len(self.proofs)} proofs and vocabulary of size {self.tokenizer.vocab_size}")
        
    def __len__(self):
        return len(self.proofs)
        
    def __getitem__(self, idx):
        proof = self.proofs[idx]
        
        # Encode theorem
        theorem = torch.tensor(self.tokenizer.encode(proof['theorem']), dtype=torch.long)
        
        # Encode proof steps
        steps = [torch.tensor(self.tokenizer.encode(step), dtype=torch.long) for step in proof['steps']]
        steps = torch.stack(steps) if steps else torch.empty((0, 0), dtype=torch.long)
        
        # Create target tokens (shifted proof steps for next-token prediction)
        target_steps = []
        for step in proof['steps']:
            tokens = self.tokenizer.encode(step)
            # Add end token to each step
            tokens = tokens + [self.tokenizer.symbol2idx[self.tokenizer.end_token]]
            target_steps.append(torch.tensor(tokens, dtype=torch.long))
        target_tokens = torch.cat(target_steps) if target_steps else torch.empty(0, dtype=torch.long)
        
        # Strategy labels
        strategy_labels = torch.tensor(proof['strategy_labels'], dtype=torch.long)
        
        # Create edge information
        num_nodes = len(proof['steps'])
        edge_index = torch.tensor(proof['edge_index'], dtype=torch.long) if num_nodes > 0 else torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.tensor(proof['edge_type'], dtype=torch.long) if num_nodes > 0 else torch.empty(0, dtype=torch.long)
        
        # Create and encode symbols
        symbols = torch.tensor(self.tokenizer.encode(" ".join(proof['symbols'])), dtype=torch.long)
        
        # Add difficulty if available
        difficulty = torch.tensor(proof.get('difficulty', 0.0), dtype=torch.float32)
        
        return {
            'symbols': symbols,
            'theorem': theorem,
            'steps': steps,
            'target_tokens': target_tokens,
            'strategy_labels': strategy_labels,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'difficulty': difficulty
        }

def pad_1d_tensors(tensors: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    """Pad a list of 1D tensors to the same length"""
    max_len = max(tensor.size(0) for tensor in tensors)
    padded = []
    for tensor in tensors:
        if tensor.size(0) < max_len:
            padding = torch.full((max_len - tensor.size(0),), pad_value,
                               dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, padding])
        padded.append(tensor)
    return torch.stack(padded)

def pad_2d_tensors(tensors: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    """Pad a list of 2D tensors to the same dimensions"""
    max_len1 = max(tensor.size(0) for tensor in tensors)
    max_len2 = max(tensor.size(1) for tensor in tensors)
    padded = []
    for tensor in tensors:
        if tensor.size(0) < max_len1 or tensor.size(1) < max_len2:
            padded_tensor = torch.full((max_len1, max_len2), pad_value,
                                     dtype=tensor.dtype, device=tensor.device)
            padded_tensor[:tensor.size(0), :tensor.size(1)] = tensor
            tensor = padded_tensor
        padded.append(tensor)
    return torch.stack(padded)

def collate_proofs(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Custom collate function to handle variable length sequences in proof data"""
    try:
        processed_batch = {}
        
        # Handle basic fields with 1D tensors
        for field in ['symbols', 'theorem', 'target_tokens']:
            if field in batch[0]:
                processed_batch[field] = pad_1d_tensors([item[field] for item in batch]).long()
                
        # Handle 2D tensors (steps)
        if 'steps' in batch[0]:
            steps = [item['steps'] for item in batch]
            if steps[0].dim() == 2:
                processed_batch['steps'] = pad_2d_tensors(steps).long()
            else:
                processed_batch['steps'] = pad_1d_tensors(steps).long()
        
        # Handle strategy labels
        if 'strategy_labels' in batch[0]:
            max_labels = max(item['strategy_labels'].size(0) for item in batch)
            padded_labels = []
            for item in batch:
                labels = item['strategy_labels']
                if labels.size(0) < max_labels:
                    padding = torch.zeros(max_labels - labels.size(0),
                                       dtype=labels.dtype,
                                       device=labels.device)
                    labels = torch.cat([labels, padding])
                padded_labels.append(labels)
            processed_batch['strategy_labels'] = torch.stack(padded_labels).long()
        
        # Handle graph-related fields
        if 'edge_index' in batch[0]:
            batch_edge_index = []
            batch_edge_type = []
            offset = 0
            for item in batch:
                edge_idx = item['edge_index']
                batch_edge_index.append(edge_idx + offset)
                if 'edge_type' in item:
                    batch_edge_type.append(item['edge_type'])
                offset += len(item['symbols']) if 'symbols' in item else len(item['theorem'])
            
            processed_batch['edge_index'] = torch.cat(batch_edge_index, dim=1).long()
            if batch_edge_type:
                processed_batch['edge_type'] = torch.cat(batch_edge_type).long()
        
        # Handle scalar values (like difficulty)
        if 'difficulty' in batch[0]:
            difficulties = [item['difficulty'] for item in batch]
            if isinstance(difficulties[0], torch.Tensor):
                processed_batch['difficulty'] = torch.stack(difficulties).float()
            else:
                processed_batch['difficulty'] = torch.tensor(difficulties, dtype=torch.float)
        
        return processed_batch
    
    except Exception as e:
        logging.error("Error in collate_proofs:")
        logging.error(f"Batch size: {len(batch)}")
        logging.error("Batch contents:")
        for i, item in enumerate(batch):
            logging.error(f"Item {i}:")
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    logging.error(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    logging.error(f"  {k}: type={type(v)}")
        raise