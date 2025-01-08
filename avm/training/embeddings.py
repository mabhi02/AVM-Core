import torch
import torch.nn as nn
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Positional Encoding for transformers."""
    
    def __init__(self, hidden_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        # Create positional encoding matrix
        pe = torch.zeros(1, max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        
        # Calculate positional encodings
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (won't be updated during backprop)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            logger.warning(f"Input sequence length {seq_len} exceeds maximum length {self.max_len}")
            x = x[:, :self.max_len, :]
            seq_len = self.max_len
            
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class MultiTypeEmbedding(nn.Module):
    """Combined embedding module that handles tokens, positions, and types."""
    
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        max_len: int = 5000,
        num_types: int = 3,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_types = num_types
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=padding_idx
        )
        
        # Token type embedding layer
        self.type_embedding = nn.Embedding(
            num_embeddings=num_types,
            embedding_dim=hidden_dim,
            padding_idx=None
        )
        
        # Positional encoding
        self.position_encoding = PositionalEncoding(
            hidden_dim=hidden_dim,
            max_len=max_len,
            dropout=dropout
        )
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.type_embedding.weight, mean=0, std=0.02)
        if self.token_embedding.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.token_embedding.padding_idx].fill_(0)

    def forward(self, tokens: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the embedding layer.
        
        Args:
            tokens: Input tensor of token ids [batch_size, seq_len]
            token_type_ids: Optional tensor of token type ids [batch_size, seq_len]
            
        Returns:
            Combined embeddings tensor [batch_size, seq_len, hidden_dim]
        """
        seq_len = tokens.size(1)
        if seq_len > self.max_len:
            logger.warning(f"Input sequence length {seq_len} exceeds maximum length {self.max_len}")
            tokens = tokens[:, :self.max_len]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, :self.max_len]
            seq_len = self.max_len
            
        # Get token embeddings
        embeddings = self.token_embedding(tokens)
        
        # Add token type embeddings if provided
        if token_type_ids is not None:
            type_embeddings = self.type_embedding(token_type_ids)
            embeddings = embeddings + type_embeddings
            
        # Apply layer normalization
        embeddings = self.layer_norm(embeddings)
        
        # Apply positional encoding
        embeddings = self.position_encoding(embeddings)
        
        return self.dropout(embeddings)

    def get_embedding_matrix(self) -> torch.Tensor:
        """Get the token embedding matrix."""
        return self.token_embedding.weight.data.clone()