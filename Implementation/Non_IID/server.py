import torch
import numpy as np
import os
import copy
import csv
import json
import matplotlib.pyplot as plt
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
    def __init__(self, num_clients=100, clients_per_round=30, rounds=40, mu=0.01, alpha=0.1, seed=1001):
        self.num_clients = num_clients
        self.rounds = rounds
        self.mu = mu
        self.alpha = alpha
        self.clients_per_round = clients_per_round
        self.seed = seed

        self.global_model = models.get_model().to(DEVICE)

        print(f"Initializing {num_clients} clients (Alpha={alpha})...")
        self.clients = [Client(client_id=i, num_clients=num_clients, alpha=alpha) 
                        for i in range(num_clients)]

        self.history = {'loss': [], 'accuracy': []}

        self.client_selection_history = {} 

        self.experiment_name = f"Alpha_{alpha}_Mu_{mu}"
        self.save_dir = os.path.join("results", self.experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        """
        The Main FL Loop.
        """
        print(f"\n--- Starting Experiment: {self.experiment_name} ---")
        
        for round_idx in range(1, self.rounds + 1):
            np.random.seed(self.seed + round_idx)

            selected_indices = np.random.choice(self.num_clients, self.clients_per_round, replace=False)
            selected_clients = [self.clients[i] for i in selected_indices]
            
            self.client_selection_history[round_idx] = selected_indices.tolist()

            print(f"Round {round_idx}/{self.rounds} - Selected {len(selected_clients)} clients.")
            
            global_weights = self.global_model.state_dict()
            
            client_updates = []
            client_sizes = []
            
            round_losses = []
            round_accuracies = []

            for client in selected_clients:
                local_weights, num_samples, metrics = client.fit(
                    global_weights, 
                    epochs=1, 
                    mu=self.mu
                )
                
                client_updates.append(local_weights)
                client_sizes.append(num_samples)

                round_losses.append(metrics['loss'] * num_samples)
                round_accuracies.append(metrics['accuracy'] * num_samples)
            
            new_global_weights = get_average_weights(client_updates, client_sizes)
            self.global_model.load_state_dict(new_global_weights)
            
            total_samples = sum(client_sizes)
            avg_loss = sum(round_losses) / total_samples
            avg_acc = sum(round_accuracies) / total_samples
            
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(avg_acc)
            
            print(f"   Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2%}")

        self.save_results()
        self.plot_metrics()

        return self.history
    
    def save_results(self):
        # Save Model
        torch.save(self.global_model.state_dict(), os.path.join(self.save_dir, "global_model.pth"))
        
        # Save CSV
        csv_path = os.path.join(self.save_dir, "metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Round', 'Loss', 'Accuracy'])
            for r, (l, a) in enumerate(zip(self.history['loss'], self.history['accuracy']), 1):
                writer.writerow([r, l, a])

        # Save Clients Logs
        json_path = os.path.join(self.save_dir, "selected_clients.json")
        with open(json_path, 'w') as f:
            json.dump(self.client_selection_history, f, indent=4)

        print(f"Data saved to {self.save_dir}")

    def plot_metrics(self):
        """
        Generates plots for Global Loss and Accuracy and saves them to a file.
        """
        rounds = range(1, len(self.history['loss']) + 1)

        final_loss = self.history['loss'][-1]
        final_acc = self.history['accuracy'][-1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))


        # Plot Loss
        ax1.plot(rounds, self.history['loss'], 'r-o')
        ax1.set_title(f'Global Loss (Mu={self.mu}, Alpha={self.alpha})')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        ax1.annotate(f"{final_loss:.2%}", 
                     (rounds[-1], final_loss), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')

        # Plot Accuracy
        ax2.plot(rounds, self.history['accuracy'], 'b-o')
        ax2.set_title(f'Global Accuracy (Mu={self.mu}, Alpha={self.alpha})')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)

        ax2.annotate(f"{final_acc:.2%}", 
                     (rounds[-1], final_acc), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')

        plt.tight_layout()

        # Save Plot
        plot_path = os.path.join(self.save_dir, "training_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    server = Server(num_clients=5, rounds=10)
    server.train()