import re
from typing import List, Dict, Union, Optional, Tuple
import torch
import json

class ProofTokenizer:
    """Tokenizer for mathematical proofs and theorems"""
    def __init__(self, vocab_file: str):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            
        self.symbol2idx = {sym: idx for idx, sym in enumerate(self.vocab['symbols'])}
        self.idx2symbol = {idx: sym for sym, idx in self.symbol2idx.items()}
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        
        # Mathematical symbols regex patterns
        self.math_patterns = [
            r'\b[a-zA-Z][0-9]*\b',  # Variables like x, y, x1, y2
            r'[+\-*/=<>≤≥≠∈∉⊆⊂∪∩∀∃¬∧∨⇒⇔]+',  # Mathematical operators
            r'\b\d+\b',  # Numbers
            r'[\(\)\[\]\{\}]',  # Brackets and parentheses
            r'"[^"]*"',  # Quoted strings
            r'\b(?:if|then|therefore|hence|thus|proof|assume|suppose|let)\b'  # Keywords
        ]
        self.math_pattern = '|'.join(self.math_patterns)
        
    def tokenize(self, text: str) -> List[str]:
        """Split text into mathematical tokens"""
        # Clean text
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Extract tokens
        tokens = []
        pos = 0
        while pos < len(text):
            match = None
            # Try to match a mathematical pattern
            for pattern in self.math_patterns:
                regex = re.compile(pattern)
                match = regex.match(text, pos)
                if match:
                    token = match.group(0)
                    tokens.append(token)
                    pos = match.end()
                    break
            
            # If no pattern matches, take the next character as a token
            if not match:
                if not text[pos].isspace():
                    tokens.append(text[pos])
                pos += 1
                
        return tokens
    
    def encode(self, text: str, max_length: Optional[int] = None) -> torch.Tensor:
        """Convert text to token indices"""
        tokens = self.tokenize(text)
        tokens = [self.start_token] + tokens + [self.end_token]
        
        # Convert to indices
        indices = [self.symbol2idx.get(token, self.symbol2idx[self.unk_token]) 
                  for token in tokens]
        
        # Pad or truncate
        if max_length is not None:
            if len(indices) < max_length:
                indices += [self.symbol2idx[self.pad_token]] * (max_length - len(indices))
            else:
                indices = indices[:max_length-1] + [self.symbol2idx[self.end_token]]
                
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices: torch.Tensor) -> str:
        """Convert token indices back to text"""
        tokens = []
        for idx in indices.tolist():
            token = self.idx2symbol[idx]
            if token in [self.pad_token, self.end_token]:
                break
            if token != self.start_token:
                tokens.append(token)
                
        # Add spaces around operators and numbers
        text = ''
        for i, token in enumerate(tokens):
            if i > 0 and not (self._is_operator(tokens[i-1]) or self._is_operator(token)):
                text += ' '
            text += token
            
        return text.strip()
    
    def _is_operator(self, token: str) -> bool:
        """Check if token is a mathematical operator"""
        return bool(re.match(r'[+\-*/=<>≤≥≠∈∉⊆⊂∪∩∀∃¬∧∨⇒⇔]', token))
        
    def batch_encode(self, texts: List[str], max_length: Optional[int] = None) -> torch.Tensor:
        """Encode a batch of texts"""
        encoded = [self.encode(text, max_length) for text in texts]
        return torch.stack(encoded)
    
    def batch_decode(self, batch_indices: torch.Tensor) -> List[str]:
        """Decode a batch of indices"""
        return [self.decode(indices) for indices in batch_indices]

    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary"""
        return len(self.symbol2idx)

    def get_special_tokens(self) -> Dict[str, str]:
        """Get the special tokens used by the tokenizer"""
        return {
            'pad': self.pad_token,
            'unk': self.unk_token,
            'start': self.start_token,
            'end': self.end_token
        }