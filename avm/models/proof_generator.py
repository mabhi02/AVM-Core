# proof_generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging
from ..utils.tokenizer import ProofTokenizer
from .strategies import (
    DirectProofStrategy,
    ContradictionStrategy,
    InductionStrategy,
    ConstructionStrategy,
    ReductionStrategy
)
from ..training.embeddings import MultiTypeEmbedding
import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch.optim as optim

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


import torch
torch.cuda.empty_cache()

PAD_TOKEN_ID = 0  # Define PAD_TOKEN_ID according to your vocabulary

class StrategyRouter(nn.Module):
    """Decides which proof strategy to use based on features."""

    def __init__(self, hidden_dim: int, num_strategies: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_strategies = num_strategies
        self.strategy_scorer = nn.Linear(hidden_dim, num_strategies)
        self.strategy_history = []  # Keep track of strategy usage

    def forward(self, features: torch.Tensor, proof_state: Optional[Dict] = None) -> torch.Tensor:
        """
        Args:
            features: shape [batch_size, hidden_dim]
            proof_state: Optional proof state information
        """
        # Get strategy scores
        scores = self.strategy_scorer(features)  # [batch_size, num_strategies]

        # Apply history penalties
        if self.strategy_history:
            for strat_idx, success in self.strategy_history[-3:]:
                if not success:
                    scores[:, strat_idx] *= 0.8

        return F.softmax(scores, dim=-1)

    def update_history(self, strategy_idx: int, success: bool):
        self.strategy_history.append((strategy_idx, success))

class AttentionVisualizer:
    """Visualizes attention patterns between decoders"""

    def __init__(self):
        self.attention_maps = {}

    def save_attention(self, name: str, attention: torch.Tensor):
        self.attention_maps[name] = attention.detach().cpu()

    def plot_attention(self, name: str, tokens: List[str], save_path: str):
        attn = self.attention_maps[name]
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens)
        plt.title(f"Attention Map: {name}")
        plt.savefig(save_path)
        plt.close()

class EnhancedDecoderBlock(nn.Module):
    """Enhanced decoder block with fixed attention mask handling"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(3)
        ])

        self.dropout = nn.Dropout(dropout)
        self.visualizer = AttentionVisualizer()

    def forward(self,
                x: torch.Tensor,
                other_decoder: torch.Tensor,
                is_causal: bool = False,
                return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            other_decoder: Other decoder output of shape [batch_size, seq_len, hidden_dim]
            is_causal: Whether to use causal masking
            return_attention: Whether to return attention weights
        """
        # Self attention
        residual = x
        x = self.norms[0](x)

        # Create attention mask only if causal
        self_attn_mask = None
        if is_causal:
            seq_len = x.size(1)
            # Create causal mask directly in the correct shape
            self_attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            self_attn_mask = self_attn_mask.to(x.device)

        x, self_attn = self.self_attention(
            query=x,
            key=x,
            value=x,
            attn_mask=self_attn_mask,
            need_weights=return_attention
        )

        if return_attention:
            self.visualizer.save_attention("self_attention", self_attn)
        x = self.dropout(x)
        x = residual + x

        # Cross attention
        residual = x
        x = self.norms[1](x)

        x, cross_attn = self.cross_attention(
            query=x,
            key=other_decoder,
            value=other_decoder,
            need_weights=return_attention
        )

        if return_attention:
            self.visualizer.save_attention("cross_attention", cross_attn)
        x = self.dropout(x)
        x = residual + x

        # Feed-forward network
        residual = x
        x = self.norms[2](x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x

        return x

class AVMCore(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        vocab_size: int = 50000,
        max_len: int = 4096,
        beam_size: int = 5,
        num_strategies: int = 5,
        vocab_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.beam_size = beam_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize tokenizer if vocab path provided
        self.tokenizer = None
        if vocab_path:
            self.tokenizer = ProofTokenizer(vocab_path)

        # Strategy router
        self.strategy_router = StrategyRouter(
            hidden_dim=hidden_dim,
            num_strategies=num_strategies
        )

        # Multi-Type Embeddings
        self.embedding = MultiTypeEmbedding(
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size,
            max_len=self.max_len,  # Ensure self.max_len is the updated value from the config
            num_types=3
        )

        # Enhanced decoders
        self.understanding_decoder = nn.ModuleList([
            EnhancedDecoderBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.generation_decoder = nn.ModuleList([
            EnhancedDecoderBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Proof strategies
        self.strategies = nn.ModuleList([
            DirectProofStrategy(hidden_dim),
            ContradictionStrategy(hidden_dim),
            InductionStrategy(hidden_dim),
            ConstructionStrategy(hidden_dim),
            ReductionStrategy(hidden_dim)
        ])

        # Output layers
        self.understanding_output = nn.Linear(hidden_dim, hidden_dim)
        self.generation_output = nn.Linear(hidden_dim, self.vocab_size)

        # Visualization
        self.attention_visualizer = AttentionVisualizer()

    def forward(self, batch: Dict[str, torch.Tensor], return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with memory-efficient processing."""
        try:
            from tqdm import tqdm
            progress = tqdm(total=100, desc="Forward Pass Detail", position=2, leave=False)

            # Memory-efficient processing settings
            chunk_size = 128  # Reduced from 256
            max_seq_len = 512  # Maximum sequence length to process

            # 1. Process inputs
            theorem = batch['theorem'][:, :max_seq_len]
            proof_steps = batch.get('steps', None)
            if proof_steps is not None:
                proof_steps = proof_steps[:, :, :max_seq_len]
            strategy_labels = batch.get('strategy_labels', None)
            
            batch_size = theorem.size(0)
            device = theorem.device
            progress.update(5)

            # 2. Create embeddings in chunks
            embeddings_list = []
            total_length = theorem.size(1)
            if proof_steps is not None:
                total_length += proof_steps.size(1) * proof_steps.size(2)
            
            for i in range(0, total_length, chunk_size):
                chunk_end = min(i + chunk_size, total_length)
                
                # Process theorem chunks
                if i < theorem.size(1):
                    chunk = theorem[:, i:min(chunk_end, theorem.size(1))]
                    chunk_type_ids = torch.zeros_like(chunk)
                    chunk_embeddings = self.embedding(chunk, chunk_type_ids)
                    embeddings_list.append(chunk_embeddings)

                # Process proof steps chunks if available
                if proof_steps is not None and i >= theorem.size(1):
                    proof_idx = (i - theorem.size(1)) // proof_steps.size(2)
                    if proof_idx < proof_steps.size(1):
                        chunk = proof_steps[:, proof_idx, :]
                        chunk_type_ids = torch.ones_like(chunk)
                        chunk_embeddings = self.embedding(chunk, chunk_type_ids)
                        embeddings_list.append(chunk_embeddings)
                
                torch.cuda.empty_cache()
                
            progress.update(20)

            # 3. Process through understanding decoder in chunks
            understanding_states = []
            for chunk_idx, chunk_embedding in enumerate(embeddings_list):
                chunk_states = chunk_embedding
                
                for layer_idx, layer in enumerate(self.understanding_decoder):
                    chunk_states = layer(
                        x=chunk_states,
                        other_decoder=chunk_states,
                        is_causal=False,
                        return_attention=return_attention
                    )
                    torch.cuda.empty_cache()
                    
                understanding_states.append(chunk_states)
                logger.info(f"Processed understanding chunk {chunk_idx + 1}/{len(embeddings_list)}")
            
            # Combine understanding states
            understanding_states = torch.cat(understanding_states, dim=1)
            progress.update(40)

            # 4. Pool understanding states
            pooled_understanding = understanding_states[:, :theorem.size(1), :].mean(dim=1)
            progress.update(10)

            # 5. Get strategy weights
            strategy_weights = self.strategy_router(pooled_understanding)
            progress.update(10)

            # 6. Generation phase with memory-efficient processing
            generation_logits = []
            
            for i in range(0, understanding_states.size(1), chunk_size):
                chunk_end = min(i + chunk_size, understanding_states.size(1))
                chunk = understanding_states[:, i:chunk_end, :]
                
                # Process chunk through generation
                with torch.cuda.amp.autocast():
                    chunk_logits = self.generation_output(chunk)
                
                # Store logits
                generation_logits.append(chunk_logits)
                
                # Clear memory
                torch.cuda.empty_cache()
            
            # Combine logits efficiently
            final_logits = torch.cat(generation_logits, dim=1)
            progress.update(15)
            
            progress.close()
            
            return {
                'understanding_features': understanding_states,
                'generation_logits': final_logits,
                'strategy_weights': strategy_weights
            }

        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            if torch.cuda.is_available():
                logger.error(f"GPU Memory at error: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            raise

    def beam_search_generate(self,
                             understanding_states: torch.Tensor,
                             initial_tokens: torch.Tensor,
                             max_length: int) -> List[Dict]:
        """Generate proofs using beam search"""
        batch_size = initial_tokens.size(0)
        device = initial_tokens.device

        # Initialize beams
        beams = [{
            'tokens': initial_tokens[i],  # Shape: (1,)
            'log_prob': 0.0,
            'hidden_states': None
        } for i in range(batch_size)]

        finished_beams = [[] for _ in range(batch_size)]
        end_token_id = self.tokenizer.symbol2idx[self.tokenizer.end_token] if self.tokenizer else 2

        for _ in range(max_length):
            all_candidates = [[] for _ in range(batch_size)]
            for idx in range(batch_size):
                beam = beams[idx]
                tokens = beam['tokens'].unsqueeze(0)  # Shape: (1, seq_len)
                # Create token type IDs
                token_type_ids = torch.full_like(tokens, fill_value=2, device=device)  # Type 2 for generated tokens
                # Embed tokens
                tokens_emb = self.embedding(tokens=tokens, token_type_ids=token_type_ids)
                # Pass through generation decoder
                generation_states = tokens_emb
                for layer in self.generation_decoder:
                    generation_states = layer(
                        x=generation_states,
                        other_decoder=understanding_states[idx:idx+1],
                        is_causal=True
                    )
                # Compute logits
                logits = self.generation_output(generation_states[:, -1, :])  # Shape: (1, vocab_size)
                log_probs = F.log_softmax(logits, dim=-1)  # Shape: (1, vocab_size)
                topk_log_probs, topk_indices = log_probs.topk(self.beam_size)

                for k in range(self.beam_size):
                    next_token = topk_indices[0, k].unsqueeze(0)
                    next_log_prob = beam['log_prob'] + topk_log_probs[0, k].item()
                    new_tokens = torch.cat([beam['tokens'], next_token], dim=0)
                    candidate = {
                        'tokens': new_tokens,
                        'log_prob': next_log_prob,
                        'hidden_states': generation_states
                    }
                    if next_token.item() == end_token_id:
                        finished_beams[idx].append(candidate)
                    else:
                        all_candidates[idx].append(candidate)

            # Select top beams
            beams = []
            for idx in range(batch_size):
                all_candidates[idx].sort(key=lambda x: x['log_prob'], reverse=True)
                beams.append(all_candidates[idx][0] if all_candidates[idx] else None)

            # If all beams are finished, break
            if all(beam is None for beam in beams):
                break

        # Return best beams
        results = []
        for idx in range(batch_size):
            if finished_beams[idx]:
                finished_beams[idx].sort(key=lambda x: x['log_prob'], reverse=True)
                results.append(finished_beams[idx][0])
            else:
                # If no beam finished with end_token, take the best incomplete one
                results.append(beams[idx])

        return results

    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate comprehensive loss with proper target handling."""
        try:
            # Forward pass
            outputs = self.forward(batch)
            
            # Get target tokens, handling potential missing keys
            if 'target_tokens' not in batch:
                logger.error("Missing target_tokens in batch")
                raise KeyError("target_tokens required for loss calculation")
            
            target_tokens = batch['target_tokens']
            
            # Get generation logits
            generation_logits = outputs['generation_logits']
            
            # Log shapes for debugging
            logger.info(f"Generation logits shape: {generation_logits.shape}")
            logger.info(f"Target tokens shape: {target_tokens.shape}")
            
            # Ensure target tokens are long type (not float)
            target_tokens = target_tokens.long()
            
            # The issue is that generation_logits.shape[1] != target_tokens.shape[1]
            # We need to either:
            # 1. Truncate the generation_logits to match target_tokens length
            # 2. Or pad the target_tokens to match generation_logits length
            
            # Option 1: Truncate generation_logits to match target_tokens length
            if generation_logits.shape[1] > target_tokens.shape[1]:
                logger.info(f"Truncating generation_logits from {generation_logits.shape[1]} to {target_tokens.shape[1]}")
                generation_logits = generation_logits[:, :target_tokens.shape[1], :]
            
            # Option 2: Pad target_tokens if needed
            elif generation_logits.shape[1] < target_tokens.shape[1]:
                logger.info(f"Truncating target_tokens from {target_tokens.shape[1]} to {generation_logits.shape[1]}")
                target_tokens = target_tokens[:, :generation_logits.shape[1]]
            
            # Reshape appropriately for cross entropy loss
            batch_size, seq_len, vocab_size = generation_logits.shape
            
            # Now the shapes should be compatible
            generation_logits = generation_logits.reshape(-1, vocab_size)
            target_tokens = target_tokens.reshape(-1)
            
            # Verify shapes before loss calculation
            logger.info(f"Reshaped generation_logits shape: {generation_logits.shape}")
            logger.info(f"Reshaped target_tokens shape: {target_tokens.shape}")
            
            # Calculate generation loss
            generation_loss = nn.CrossEntropyLoss(ignore_index=0)(
                generation_logits,
                target_tokens
            )
            
            # Strategy selection loss
            strategy_loss = torch.tensor(0.0, device=self.device)
            if 'strategy_labels' in batch and 'strategy_weights' in outputs:
                strategy_weights = outputs['strategy_weights']
                strategy_labels = batch['strategy_labels']
                
                # Log shapes for debugging
                logger.info(f"Strategy weights shape: {strategy_weights.shape}")
                logger.info(f"Strategy labels shape: {strategy_labels.shape}")
                
                # Handle multi-dimensional strategy labels
                if strategy_labels.dim() > 1:
                    # Take the first strategy label for each batch item
                    strategy_labels = strategy_labels[:, 0]
                    logger.info(f"Using first strategy label, new shape: {strategy_labels.shape}")
                
                # Ensure strategy labels are long
                strategy_labels = strategy_labels.long()
                
                strategy_loss = nn.CrossEntropyLoss()(
                    strategy_weights,
                    strategy_labels
                )
            
            # Total loss
            total_loss = generation_loss + 0.5 * strategy_loss
            
            return {
                'loss': total_loss,
                'generation_loss': generation_loss,
                'strategy_loss': strategy_loss
            }
            
        except Exception as e:
            logger.error(f"Error in loss calculation: {str(e)}")
            logger.error("Batch contents:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    logger.error(f"{k}: shape={v.shape}, dtype={v.dtype}")
            raise

    def generate_proof(self, theorem_tokens: torch.Tensor,
                       max_steps: int = 50,
                       return_attention: bool = False) -> Dict:
        """Generate proof with visualization and strategy information"""
        self.eval()
        with torch.no_grad():
            # Prepare batch
            batch = {'theorem': theorem_tokens}
            outputs = self.forward(
                batch,
                return_attention=return_attention
            )

            understanding_states = outputs['understanding_features']
            batch_size = theorem_tokens.size(0)

            # Start tokens for generation
            initial_tokens = torch.full((batch_size, 1), self.tokenizer.symbol2idx[self.tokenizer.start_token], dtype=torch.long, device=self.device)

            # Beam search to generate proofs
            beams = self.beam_search_generate(
                understanding_states,
                initial_tokens,
                max_length=max_steps
            )

            # Collect results
            generated_tokens = torch.stack([beam['tokens'] for beam in beams], dim=0)  # (batch_size, seq_len)
            confidence_scores = [beam['log_prob'] for beam in beams]

            # Prepare the output dictionary
            results = {
                'proof_tokens': generated_tokens,
                'confidence_scores': confidence_scores,
                'understanding_features': understanding_states,
                'strategy_weights': outputs['strategy_weights'],
                'attention_maps': outputs.get('attention_maps', {}),
                'alternative_proofs': [],  # Collect alternative proofs if needed
                'intermediate_steps': [],  # Update if intermediate steps are maintained
                'strategy_analysis': {
                    'strategy_changes': len(self.strategy_router.strategy_history),
                    'successful_strategies': sum(1 for _, success in self.strategy_router.strategy_history if success),
                    'strategy_distribution': outputs['strategy_weights'].cpu().numpy()
                }
            }

            return results
