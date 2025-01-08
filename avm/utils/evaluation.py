import torch
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

class ProofEvaluator:
    """Evaluation metrics for proof generation"""
    def __init__(self):
        self.metrics_history = {
            'step_accuracy': [],
            'strategy_accuracy': [],
            'proof_validity': [],
            'completeness': []
        }
        
    def evaluate_proof(self, 
                      generated_proof: Dict,
                      target_proof: Dict) -> Dict[str, float]:
        """Evaluate a generated proof against target proof"""
        metrics = {}
        
        # Evaluate proof steps
        metrics['step_accuracy'] = self._evaluate_steps(
            generated_proof['steps'],
            target_proof['steps']
        )
        
        # Evaluate strategy selection
        metrics['strategy_accuracy'] = self._evaluate_strategy(
            generated_proof['used_strategies'],
            target_proof['target_strategies']
        )
        
        # Evaluate overall proof validity
        metrics['proof_validity'] = self._evaluate_validity(
            generated_proof['steps'],
            target_proof['theorem']
        )
        
        # Evaluate completeness
        metrics['completeness'] = self._evaluate_completeness(
            generated_proof['steps'],
            target_proof['theorem']
        )
        
        # Update history
        for metric, value in metrics.items():
            self.metrics_history[metric].append(value)
            
        return metrics
    
    def _evaluate_steps(self,
                       generated_steps: List[str],
                       target_steps: List[str]) -> float:
        """Evaluate accuracy of generated proof steps"""
        # Convert steps to sentence embeddings for semantic comparison
        gen_embeddings = self._get_embeddings(generated_steps)
        target_embeddings = self._get_embeddings(target_steps)
        
        # Calculate cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            gen_embeddings.unsqueeze(1),
            target_embeddings.unsqueeze(0),
            dim=-1
        )
        
        return similarities.max(dim=1)[0].mean().item()
    
    def _evaluate_strategy(self,
                         used_strategies: List[str],
                         target_strategies: List[str]) -> float:
        """Evaluate strategy selection accuracy"""
        used_set = set(used_strategies)
        target_set = set(target_strategies)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            [s in target_set for s in used_set],
            [True] * len(used_set),
            average='binary'
        )
        
        return f1
    
    def _evaluate_validity(self,
                         steps: List[str],
                         theorem: str) -> float:
        """Evaluate logical validity of the proof"""
        # This would require a formal logic checker
        # For now, using a simple heuristic
        required_keywords = ['therefore', 'thus', 'hence', 'consequently']
        has_conclusion = any(
            keyword in steps[-1].lower() 
            for keyword in required_keywords
        )
        
        return float(has_conclusion)
    
    def _evaluate_completeness(self,
                             steps: List[str],
                             theorem: str) -> float:
        """Evaluate if proof fully proves the theorem"""
        # Simple heuristic based on final step matching theorem
        final_step = steps[-1].lower()
        theorem = theorem.lower()
        
        return float(any(word in final_step for word in theorem.split()))
    
    def _get_embeddings(self, sentences: List[str]) -> torch.Tensor:
        """Convert sentences to embeddings for comparison"""
        # In practice, you would use a pre-trained model like BERT
        # This is a placeholder implementation
        return torch.randn(len(sentences), 768)  # 768 is BERT dimension
    
    def plot_metrics_history(self, save_path: str = 'metrics_history.png'):
        """Plot the history of evaluation metrics"""
        plt.figure(figsize=(12, 6))
        
        for metric, values in self.metrics_history.items():
            plt.plot(values, label=metric)
            
        plt.xlabel('Evaluation Step')
        plt.ylabel('Score')
        plt.title('Evaluation Metrics History')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(save_path)
        plt.close()