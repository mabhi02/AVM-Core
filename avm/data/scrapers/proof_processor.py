import json
import re
import numpy as np
from typing import List, Dict
from collections import Counter
import os

class ProofProcessor:
    def __init__(self):
        self.special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        self.symbol2idx = {}
        self.vocab = {'symbols': []}
        
    def process_proofs(self, input_file: str, output_dir: str, vocab_size: int = 5000):
        """Process raw StackExchange proofs into training format"""
        # Load raw proofs with UTF-8 encoding
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                raw_proofs = json.load(f)
        except UnicodeDecodeError:
            print("Trying alternate encoding...")
            with open(input_file, 'r', encoding='utf-8-sig') as f:
                raw_proofs = json.load(f)
            
        processed_proofs = []
        all_symbols = Counter()
        
        print(f"Processing {len(raw_proofs)} proofs...")
        
        for proof in raw_proofs:
            try:
                # Extract theorem and proof
                theorem = self._clean_text(proof['theorem'])
                proof_text = self._clean_text(proof['proof'])
                
                # Split into steps
                steps = self._split_into_steps(proof_text)
                
                # Extract symbols and build vocabulary
                symbols = self._extract_symbols(theorem + " " + proof_text)
                all_symbols.update(symbols)
                
                # Extract relations between steps
                relations = self._extract_relations(steps)
                
                # Estimate difficulty based on length and complexity
                difficulty = self._estimate_difficulty(theorem, steps)
                
                # Identify proof strategies
                strategy_labels = self._identify_strategies(steps)
                
                processed_proofs.append({
                    'theorem': theorem,
                    'symbols': symbols,
                    'steps': steps,
                    'relations': relations,
                    'strategy_labels': strategy_labels,
                    'difficulty': difficulty
                })
                
            except Exception as e:
                print(f"Error processing proof: {e}")
                continue
        
        print(f"Successfully processed {len(processed_proofs)} proofs")
        
        # Build vocabulary
        self._build_vocabulary(all_symbols, vocab_size)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Split into train and validation sets (90-10 split)
        np.random.seed(42)
        indices = np.random.permutation(len(processed_proofs))
        split_idx = int(len(processed_proofs) * 0.9)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_proofs = [processed_proofs[i] for i in train_indices]
        val_proofs = [processed_proofs[i] for i in val_indices]
        
        # Save processed data
        with open(os.path.join(output_dir, 'train_proofs.json'), 'w', encoding='utf-8') as f:
            json.dump(train_proofs, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(output_dir, 'val_proofs.json'), 'w', encoding='utf-8') as f:
            json.dump(val_proofs, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(output_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            
        print(f"Saved {len(train_proofs)} training proofs and {len(val_proofs)} validation proofs")
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove HTML
        text = re.sub(r'<[^>]+>', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize math symbols
        text = text.replace('\\', '')
        return text.strip()
        
    def _split_into_steps(self, proof_text: str) -> List[str]:
        """Split proof into logical steps"""
        # Split on common step markers
        markers = r'(?:Therefore|Thus|Hence|So|Next|First|Second|Finally|Note that|Observe that|Consider|Let)'
        steps = re.split(f'(?:{markers}),?', proof_text)
        steps = [s.strip() for s in steps if s.strip()]
        if not steps:  # If no clear steps found, split by sentences
            steps = [s.strip() for s in re.split(r'[.!?]+', proof_text) if s.strip()]
        return steps
        
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract mathematical symbols and terms"""
        # Extract potential mathematical symbols
        symbols = re.findall(r'[a-zA-Z0-9+\-*/=<>≤≥≠∈∉⊆⊂∪∩∀∃¬∧∨⇒⇔]+', text)
        return symbols
        
    def _extract_relations(self, steps: List[str]) -> List[Dict]:
        """Extract relationships between proof steps"""
        relations = []
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                # Look for references between steps
                if any(marker in steps[j].lower() for marker in ['previous', 'above', 'from this']):
                    relations.append({
                        'from_idx': i,
                        'to_idx': j,
                        'type_idx': 0  # Default to basic dependency
                    })
        return relations
        
    def _estimate_difficulty(self, theorem: str, steps: List[str]) -> float:
        """Estimate proof difficulty"""
        # Simple heuristic based on length and complexity
        total_length = len(theorem) + sum(len(step) for step in steps)
        num_steps = len(steps)
        symbol_complexity = len(set(self._extract_symbols(theorem)))
        
        difficulty = (total_length / 500) * 0.4 + (num_steps / 10) * 0.3 + (symbol_complexity / 20) * 0.3
        return min(max(difficulty, 0.0), 1.0)
        
    def _identify_strategies(self, steps: List[str]) -> List[int]:
        """Identify proof strategies used"""
        strategies = []
        strategy_keywords = {
            0: ['contradiction', 'suppose', 'assume'],
            1: ['induction', 'base case', 'inductive'],
            2: ['direct', 'clearly', 'obviously'],
            3: ['construction', 'define', 'let us construct']
        }
        
        for step in steps:
            step_lower = step.lower()
            found = False
            for strategy_id, keywords in strategy_keywords.items():
                if any(keyword in step_lower for keyword in keywords):
                    strategies.append(strategy_id)
                    found = True
                    break
            if not found:
                strategies.append(2)  # Default to direct proof
        
        return strategies
        
    def _build_vocabulary(self, symbol_counts: Counter, vocab_size: int):
        """Build vocabulary from collected symbols"""
        # Add special tokens
        self.vocab['symbols'] = self.special_tokens.copy()
        
        # Add most common symbols
        most_common = [symbol for symbol, _ in symbol_counts.most_common(vocab_size - len(self.special_tokens))]
        self.vocab['symbols'].extend(most_common)
        
        # Update symbol to index mapping
        self.symbol2idx = {sym: idx for idx, sym in enumerate(self.vocab['symbols'])}

if __name__ == "__main__":
    print("Starting proof processing...")
    processor = ProofProcessor()
    processor.process_proofs(
        input_file='../../data/math_proofs.json',
        output_dir='../../data/processed'
    )
    print("Processing completed!")