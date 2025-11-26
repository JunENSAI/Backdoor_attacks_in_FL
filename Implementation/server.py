import torch
import numpy as np
import copy
import models
from client import Client

# --- Configuration ---
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
    
    # 1. Initialize the aggregator with the structure of the first update
    # We need a 'zero' model to start adding things to.
    avg_weights = copy.deepcopy(clients_updates[0])
    
    # Set every parameter to 0 initially
    for key in avg_weights.keys():
        avg_weights[key] = torch.zeros_like(avg_weights[key], dtype=torch.float32)
        
    # 2. Loop through every client's update
    for client_weights, client_size in zip(clients_updates, client_dataset_sizes):
        
        # Calculate the "Vote Power" of this client
        # If a client has more data, their update matters more.
        contribution_ratio = client_size / total_data_points
        
        # Add their weighted contribution to the global sum
        for key in avg_weights.keys():
            avg_weights[key] += client_weights[key] * contribution_ratio
            
    return avg_weights

class Server:
    def __init__(self, num_clients=10, rounds=5):
        self.num_clients = num_clients
        self.rounds = rounds
        self.global_model = models.get_model().to(DEVICE)
        
        # Simulation: Create the clients (In real life, they connect to us)
        self.clients = [Client(client_id=i) for i in range(num_clients)]

    def train(self):
        """
        The Main FL Loop.
        """
        print(f"--- Starting Federated Learning on {DEVICE} ---")
        
        for round_idx in range(1, self.rounds + 1):
            print(f"\n--- Round {round_idx}/{self.rounds} ---")
            
            # 1. Select Clients
            # For this simulation, we use ALL clients every round.
            # In real FL, you might pick a random sample (e.g., 10 out of 1000).
            selected_clients = self.clients 
            
            global_weights = self.global_model.state_dict()
            
            client_updates = []
            client_sizes = []
            
            # 2. Communicate with Clients
            for client in selected_clients:
                # A. Transmit Global Weights -> Client
                # B. Client Trains (fit) locally
                # C. Receive New Weights <- Client
                local_weights, num_samples = client.fit(global_weights, epochs=1)
                
                # Store the results
                client_updates.append(local_weights)
                client_sizes.append(num_samples)
            
            # 3. Aggregation (FedAvg)
            # Combine all client knowledge into one new global model
            new_global_weights = get_average_weights(client_updates, client_sizes)
            
            # 4. Update the Server's Model
            self.global_model.load_state_dict(new_global_weights)
            
            # 5. Optional: Evaluate Global Model (on a server-side test set if available)
            # Here we just print a success message.
            print("Round Complete. Global Model Updated.")

# This code would be triggered by main.py
if __name__ == "__main__":
    server = Server(num_clients=5, rounds=3)
    server.train()