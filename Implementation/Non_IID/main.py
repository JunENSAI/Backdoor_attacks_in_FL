import argparse
import torch
import random
import numpy as np
from server import Server

def main():    
    parser = argparse.ArgumentParser(description="Federated Learning Simulation")
    
    parser.add_argument('--rounds', type=int, default=8, 
                        help="Number of Global Communication Rounds")
    
    parser.add_argument('--clients', type=int, default=9, 
                        help="Number of Clients in the federation")
    
    parser.add_argument('--seed', type=int, default=1001, 
                        help="Random seed for reproducibility")
    
    parser.add_argument('--alpha', type=float, default=0.01, 
                        help="Dirichlet Alpha (0.1=Non-IID, 10=IID)")
    
    parser.add_argument('--mu', type=float, default=0.01, 
                        help="FedProx Proximal Term (0.0 = FedAvg)")
    
    args = parser.parse_args()
    
    # Initialize server with mu
    server = Server(num_clients=args.clients, rounds=args.rounds, mu=args.mu)
    server.train()
if __name__ == "__main__":
    main()