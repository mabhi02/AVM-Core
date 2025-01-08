import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict
import torch
import seaborn as sns
from pathlib import Path

class ProofVisualizer:
    """Visualization tools for proof generation"""
    def __init__(self, save_dir: str = 'visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_proof_graph(self, 
                        symbols: List[str], 
                        relations: List[Dict],
                        filename: str = 'proof_graph.png'):
        """Plot the graph of mathematical symbols and their relations"""
        G = nx.Graph()
        
        # Add nodes
        for symbol in symbols:
            G.add_node(symbol)
            
        # Add edges
        for relation in relations:
            G.add_edge(relation['from'], relation['to'], 
                      type=relation.get('type', 'undefined'))
            
        # Create layout
        pos = nx.spring_layout(G)
        
        # Plot
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=1500, font_size=10)
        
        # Add edge labels
        edge_labels = nx.get_edge_attributes(G, 'type')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.savefig(self.save_dir / filename)
        plt.close()
        
    def plot_training_progress(self,
                             metrics: Dict[str, List[float]],
                             filename: str = 'training_progress.png'):
        """Plot training metrics over time"""
        plt.figure(figsize=(12, 6))
        
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
            
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(self.save_dir / filename)
        plt.close()
        
    def plot_attention_weights(self,
                             attention_weights: torch.Tensor,
                             tokens: List[str],
                             filename: str = 'attention.png'):
        """Plot attention weights heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights.cpu().numpy(),
                   xticklabels=tokens,
                   yticklabels=tokens,
                   cmap='Blues')
        
        plt.title('Attention Weights')
        plt.tight_layout()
        plt.savefig(self.save_dir / filename)
        plt.close()
        
    def plot_strategy_distribution(self,
                                 strategy_usage: Dict[str, int],
                                 filename: str = 'strategy_dist.png'):
        """Plot distribution of proof strategies used"""
        plt.figure(figsize=(10, 6))
        
        strategies = list(strategy_usage.keys())
        counts = list(strategy_usage.values())
        
        plt.bar(strategies, counts)
        plt.xticks(rotation=45)
        plt.xlabel('Strategy')
        plt.ylabel('Usage Count')
        plt.title('Proof Strategy Distribution')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / filename)
        plt.close()