import argparse
import torch
import random
import numpy as np
from server import Server

def set_seed(seed=10001):
    """
    Sets the random seed for reproducibility.
    This ensures that every time you run the code, the 'random' initialization
    of weights and data splits is exactly the same. 
    Crucial for debugging!
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensure determinstic behavior in CuDNN (for Nvidia GPUs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    parser = argparse.ArgumentParser(description="Federated Learning Simulation")
    
    parser.add_argument('--rounds', type=int, default=20, 
                        help="Number of Global Communication Rounds")
    
    parser.add_argument('--clients', type=int, default=5, 
                        help="Number of Clients in the federation")
    
    parser.add_argument('--seed', type=int, default=10001, 
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    print(f"--- Configuration: {args.clients} Clients, {args.rounds} Rounds ---")
    set_seed(args.seed)

    server = Server(num_clients=args.clients, rounds=args.rounds)

    server.train()

if __name__ == "__main__":
    main()