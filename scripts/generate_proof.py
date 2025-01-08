import argparse
import torch
from avm.models.proof_generator import AVMCore
from avm.utils.visualization import ProofVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Generate proof using AVM-CORE')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--theorem', type=str, required=True,
                      help='Theorem to prove')
    parser.add_argument('--output', type=str, default='generated_proof.png',
                      help='Output file for visualization')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load model
    model = AVMCore.load_from_checkpoint(args.checkpoint)
    model.eval()
    
    # Generate proof
    proof = model.generate_proof(args.theorem)
    
    # Print proof
    print("\nGenerated Proof:")
    for i, step in enumerate(proof['steps'], 1):
        print(f"{i}. {step}")
        
    # Visualize
    visualizer = ProofVisualizer()
    visualizer.plot_proof_graph(
        proof['symbols'],
        proof['relations'],
        filename=args.output
    )
    
    print(f"\nProof visualization saved to {args.output}")

if __name__ == '__main__':
    main()