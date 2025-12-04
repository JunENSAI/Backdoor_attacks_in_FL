import torch
import numpy as np
import copy
import models
from client import Client
import matplotlib.pyplot as plt

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
    def __init__(self, num_clients=10, rounds=5, mu=0.01, alpha=0.1):
        self.num_clients = num_clients
        self.rounds = rounds
        self.mu = mu
        self.alpha = alpha
        self.global_model = models.get_model().to(DEVICE)
        
        self.clients = [Client(client_id=i, num_clients=num_clients, alpha=alpha) 
                        for i in range(num_clients)]

        self.history = {'loss': [], 'accuracy': []}

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
            
            print(f"Round {round_idx} Complete. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2%}")

        self.plot_metrics()

    def plot_metrics(self):
        """
        Generates plots for Global Loss and Accuracy and saves them to a file.
        """
        rounds = range(1, self.rounds + 1)

        final_loss = self.history['loss'][-1]
        final_acc = self.history['accuracy'][-1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(rounds, self.history['loss'], color='tab:red', marker='o', linestyle='-')
        ax1.set_title(f'Global Loss (Final: {final_loss:.4f})')
        ax1.set_xlabel('Communication Round')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        ax2.plot(rounds, self.history['accuracy'], color='tab:blue', marker='o', linestyle='-')
        ax2.set_title(f'Global Accuracy (Final: {final_acc:.2%})')
        ax2.set_xlabel('Communication Round')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1.05])
        ax2.grid(True)

        ax2.annotate(f"{final_acc:.2%}", 
                     (rounds[-1], final_acc), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')
        
        plt.tight_layout()
        output_file = 'fl_iid_results.png'
        plt.savefig(output_file)
        print(f"\nPlots saved to {output_file}. Final Accuracy: {final_acc:.2%}")
        plt.show()

if __name__ == "__main__":
    server = Server(num_clients=5, rounds=10)
    server.train()