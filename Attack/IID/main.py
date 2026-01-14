# pylint: disable=missing-function-docstring
import sys
import argparse
import random
import torch
import numpy as np
from server import Server

def set_seed(seed=10001):
    """
    Sets the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Backdoor Simulation")
    
    # PROFESSOR'S GOAL: Increase rounds to 100+
    parser.add_argument('--rounds', type=int, default=100, 
                        help="Number of Global Communication Rounds")
    
    parser.add_argument('--clients', type=int, default=100, 
                        help="Total pool of potential clients")
    
    parser.add_argument('--sample', type=int, default=30, 
                        help="Number of clients selected per round (m)")
    
    # PROFESSOR'S GOAL: Alpha parameter for malicious probability
    parser.add_argument('--alpha', type=float, default=0.2, 
                        help="Probability (0 to 1) that a selected client is malicious")
    
    parser.add_argument('--seed', type=int, default=1509, 
                        help="Random seed for reproducibility")

    if 'ipykernel' in sys.modules or 'colab' in sys.modules:
        args = parser.parse_args(args=[]) 
    else:
        args = parser.parse_args()

    # Log the key attack parameters
    print(f"--- Configuration ---")
    print(f"Total Clients (N): {args.clients}")
    print(f"Clients per Round (m): {args.sample}")
    print(f"Malicious Probability (Alpha): {args.alpha}")
    print(f"Total Rounds: {args.rounds}")
    print(f"----------------------")

    set_seed(args.seed)

    # Initialize server with the new alpha parameter
    server = Server(
        num_clients=args.clients, 
        clients_per_round=args.sample, 
        rounds=args.rounds,
        alpha=args.alpha
    )

    server.train()

if __name__ == "__main__":
    main()