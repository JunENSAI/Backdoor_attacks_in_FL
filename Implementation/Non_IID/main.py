import argparse
import torch
import random
import numpy as np
from server import Server

def set_seed(seed):
    """Fixe les graines aléatoires pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():    
    parser = argparse.ArgumentParser(description="Federated Learning Simulation")
    
    parser.add_argument('--rounds', type=int, default=30, 
                        help="Number of Global Communication Rounds")
    
    parser.add_argument('--clients', type=int, default=10, 
                        help="Number of Clients in the federation")
    
    parser.add_argument('--seed', type=int, default=3001, 
                        help="Random seed for reproducibility")
    
    parser.add_argument('--alpha', type=float, default=0.1, 
                        help="Dirichlet Alpha (0.1=Non-IID, 10=IID)")
    
    parser.add_argument('--mu', type=float, default=0.01, 
                        help="FedProx Proximal Term (0.0 = FedAvg)")
    
    args = parser.parse_args()

    set_seed(args.seed)
    
    print(f"--- Configuration: Alpha={args.alpha}, Mu={args.mu}, Seed={args.seed} ---")

    server = Server(num_clients=args.clients, rounds=args.rounds, mu=args.mu, alpha=args.alpha)
    server.train()

if __name__ == "__main__":
    main()