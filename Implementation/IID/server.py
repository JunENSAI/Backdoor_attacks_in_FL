import torch
import numpy as np
import copy
import random
import csv
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
    def __init__(self, num_clients=100, clients_per_round=30, rounds=40):
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.rounds = rounds
        self.global_model = models.get_model().to(DEVICE)
        
        self.clients = [Client(client_id=i, total_clients=num_clients) for i in range(num_clients)]

        self.history = {'loss': [], 'accuracy': []}

    def train(self):
        """
        The Main FL Loop.
        """
        print(f"--- Starting Federated Learning on {DEVICE} ---")
        
        # On ouvre DEUX fichiers : 
        # 1. Le CSV pour les métriques (Loss/Acc)
        # 2. Le TXT pour l'historique des clients (ce que vous avez demandé)
        with open('fl_logs.csv', mode='w', newline='') as log_file, \
             open('client_selection_history.txt', mode='w') as history_file:
            
            # Préparation du CSV
            writer = csv.writer(log_file)
            writer.writerow(['Round', 'Average Loss', 'Average Accuracy'])

            # Préparation du fichier historique (Optionnel : ajouter un titre)
            history_file.write(f"Historique de selection des clients ({self.clients_per_round} par round)\n")
            history_file.write("===================================================\n")

            for round_idx in range(1, self.rounds + 1):
                print(f"\n--- Round {round_idx}/{self.rounds} ---")

                # 1. Sélection aléatoire
                selected_clients = random.sample(self.clients, self.clients_per_round)
                
                # --- NOUVEAU : Récupérer les IDs et les écrire dans le fichier ---
                selected_ids = [c.client_id for c in selected_clients]
                
                # Format demandé : "Round 1 : [12, 45, ...]"
                history_line = f"Round {round_idx} : {selected_ids}\n"
                history_file.write(history_line)
                # -----------------------------------------------------------------

                global_weights = self.global_model.state_dict()
                
                client_updates = []
                client_sizes = []
                round_losses = []
                round_accuracies = []

                for client in selected_clients:
                    local_weights, num_samples, metrics = client.fit(global_weights, epochs=1)
                    
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
                
                # Sauvegarde CSV
                writer.writerow([round_idx, avg_loss, avg_acc])

        self.plot_metrics()
        
        # Sauvegarde du modèle final
        torch.save(self.global_model.state_dict(), "global_model.pth")
        print("\nSauvegardes effectuees :")
        print("- Modele : global_model.pth")
        print("- Logs CSV : fl_logs.csv")
        print("- Historique Clients : client_selection_history.txt")
    
    def plot_metrics(self):
        """
        Génère les courbes et affiche la précision finale sur l'image.
        """
        rounds = range(1, self.rounds + 1)

        final_loss = self.history['loss'][-1]
        final_acc = self.history['accuracy'][-1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(rounds, self.history['loss'], color='r', linestyle='-')
        ax1.set_title(f'Global Loss (Final: {final_loss:.4f})')
        ax1.set_xlabel('Communication Round')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        ax2.plot(rounds, self.history['accuracy'], color='b', linestyle='-')
        ax2.set_title(f'Global Accuracy (Final: {final_acc:.2%})')
        ax2.set_xlabel('Communication Round')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1]) 
        ax2.grid(True)
 
        ax2.annotate(f"{final_acc:.2%}", 
                     (rounds[-1], final_acc), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')
        
        plt.tight_layout()
        plt.savefig('fl_iid_results.png')
        print(f"\nPlots saved. Final Accuracy: {final_acc:.2%}")
        plt.show()