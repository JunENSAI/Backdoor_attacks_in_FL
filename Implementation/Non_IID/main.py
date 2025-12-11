import argparse
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from server import Server

def set_seed(seed):
    """Fixe les graines aléatoires pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_comparison(all_results, alpha, save_folder="results"):
    """
    Plots a comparison of FedAvg (Mu=0) vs FedProx (Mu>0) on the same graph.
    """
    plt.figure(figsize=(10, 6))
    
    # Iterate through different Mu results
    for mu, history in all_results.items():
        rounds = range(1, len(history['accuracy']) + 1)
        label = "FedAvg (mu=0)" if mu == 0.0 else f"FedProx (mu={mu})"
        plt.plot(rounds, history['accuracy'], marker='o', label=label)

    plt.title(f'Comparative Accuracy: FedProx vs FedAvg (Alpha={alpha})')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save comparison in the main results folder
    filename = os.path.join(save_folder, f"Comparison_Alpha_{alpha}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Comparison plot saved: {filename}")

def main():
    # --- Configuration ---
    ROUNDS = 40           
    NUM_CLIENTS = 100     
    CLIENTS_PER_ROUND = 30 
    SEED = 1509
    
    # Grid Search Values
    mu_values = [0.0, 0.01, 0.1, 0.4]  # 0.0 is FedAvg
    alpha_values = [0.1, 0.5, 10]
    
    set_seed(SEED)
    
    print(f"--- Starting Grid Search ---")
    
    for alpha in alpha_values:
        results_for_alpha = {} 
        
        for mu in mu_values:
            print(f"\n> Running: Alpha={alpha}, Mu={mu}")
            
            server = Server(
                num_clients=NUM_CLIENTS, 
                clients_per_round=CLIENTS_PER_ROUND, 
                rounds=ROUNDS, 
                mu=mu, 
                alpha=alpha,
                seed=SEED
            )

            history = server.train()
            
            results_for_alpha[mu] = history
        
        plot_comparison(results_for_alpha, alpha)

if __name__ == "__main__":
    main()