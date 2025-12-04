import torch
import numpy as np
import copy
import models
from client import Client

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_average_weights(clients_updates, client_dataset_sizes):
    """
    The Heart of FedAvg (Federated Averaging).
    
    Formula: W_global = Sum( (n_k / n_total) * W_k )
    
    Args:
        clients_updates: A list of state_dicts (weights) from each client.
        client_dataset_sizes: A list of integers (how many images each client had).
    """
    total_data_points = sum(client_dataset_sizes)
    
    avg_weights = copy.deepcopy(clients_updates[0])

    for key in avg_weights.keys():
        avg_weights[key] = torch.zeros_like(avg_weights[key], dtype=torch.float32)
        
    for client_weights, client_size in zip(clients_updates, client_dataset_sizes):

        contribution_ratio = client_size / total_data_points
        
        for key in avg_weights.keys():
            avg_weights[key] += client_weights[key] * contribution_ratio
            
    return avg_weights

class Server:
    def __init__(self, num_clients=10, rounds=5,mu=0.01):
        self.num_clients = num_clients
        self.rounds = rounds
        self.mu=mu
        self.global_model = models.get_model().to(DEVICE)
        
        self.clients = [Client(client_id=i) for i in range(num_clients)]

    def train(self):
        """
        The Main FL Loop.
        """
        print(f"--- Starting Federated Learning on {DEVICE} ---")
        
        for round_idx in range(1, self.rounds + 1):
            print(f"\n--- Round {round_idx}/{self.rounds} ---")

            selected_clients = self.clients 
            
            global_weights = self.global_model.state_dict()
            
            client_updates = []
            client_sizes = []
            
            mu_factor = 0.01

            for client in selected_clients:

                local_weights, num_samples = client.fit(global_weights, epochs=1, mu=mu_factor)
                
                client_updates.append(local_weights)
                client_sizes.append(num_samples)
            
            new_global_weights = get_average_weights(client_updates, client_sizes)
   
            self.global_model.load_state_dict(new_global_weights)
            
            print("Round Complete. Global Model Updated.")

if __name__ == "__main__":
    server = Server(num_clients=5, rounds=3)
    server.train()

    