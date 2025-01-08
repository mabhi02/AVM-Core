import argparse
import torch
from pathlib import Path
from avm.models.proof_generator import AVMCore
from avm.utils.evaluation import ProofEvaluator
from avm.data.dataset import ProofDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate AVM-CORE')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                      help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save evaluation results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load model
    checkpoint = torch.load(args.checkpoint)
    model = AVMCore.load_from_checkpoint(checkpoint)
    model.eval()
    
    # Load test data
    test_dataset = ProofDataset(args.test_data)
    
    # Initialize evaluator
    evaluator = ProofEvaluator()
    
    # Evaluate
    results = []
    for theorem, target_proof in test_dataset:
        generated_proof = model.generate_proof(theorem)
        metrics = evaluator.evaluate_proof(generated_proof, target_proof)
        results.append(metrics)
        
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    evaluator.plot_metrics_history(output_dir / 'evaluation_metrics.png')
    
    # Print summary
    print("\nEvaluation Results:")
    for metric in results[0].keys():
        avg_value = sum(r[metric] for r in results) / len(results)
        print(f"{metric}: {avg_value:.4f}")

if __name__ == '__main__':
    main()