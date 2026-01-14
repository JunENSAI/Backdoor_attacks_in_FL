# pylint: disable=missing-function-docstring
import csv
import random
import copy
import torch
import numpy as np # Added for alpha logic
import matplotlib.pyplot as plt
from util import evaluate_backdoor,evaluate_asr_filtered
#import Attack.IID.util_ignore as util_ignore
import models
from client import Client, MaliciousClient

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_average_weights(clients_updates, client_dataset_sizes):
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
    def __init__(self, num_clients=100, clients_per_round=30, rounds=40, alpha=0.2):
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.rounds = rounds
        self.alpha = alpha # Probability that a selected client is malicious
        self.global_model = models.get_model().to(DEVICE)
        
        # We no longer pre-assign roles. All are base Clients initially.
        print(f"Initializing {num_clients} Potential Clients with Alpha={alpha}...")
        self.clients = [Client(client_id=i, total_clients=num_clients) for i in range(num_clients)]

        self.history = {'loss': [], 'accuracy': [], 'asr_brut': [], 'asr_filtre': []}

    def train(self):
        print(f"--- Starting Federated Learning (IID) with {self.alpha*100}% Malicious Probability ---")

        with open('fl_logs.csv', mode='w', newline='', encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(['Round', 'Average Loss', 'Average Accuracy', 'Backdoor ASR'])

            for round_idx in range(1, self.rounds + 1):
                # Step 1: Sample clients
                selected_indices = random.sample(range(self.num_clients), self.clients_per_round)
                
                global_weights = self.global_model.state_dict()
                client_updates = []
                client_sizes = []
                round_losses = []
                round_accuracies = []

                # Step 2: Dynamically assign Malicious vs Benign based on alpha
                for idx in selected_indices:
                    # If random roll is less than alpha, this client attacks this round
                    if np.random.rand() < self.alpha:
                        # Temporary malicious behavior for this round
                        attacker = MaliciousClient(client_id=idx, total_clients=self.num_clients)
                        local_weights, num_samples, metrics = attacker.fit(global_weights, epochs=1)
                    else:
                        # Standard benign behavior
                        local_weights, num_samples, metrics = self.clients[idx].fit(global_weights, epochs=1)
                    
                    client_updates.append(local_weights)
                    client_sizes.append(num_samples)
                    round_losses.append(metrics['loss'] * num_samples)
                    round_accuracies.append(metrics['accuracy'] * num_samples)

                # Step 3: Global Aggregation (FedAvg)
                new_global_weights = get_average_weights(client_updates, client_sizes)
                self.global_model.load_state_dict(new_global_weights)
                
                # Step 4: Metric Tracking
                total_samples = sum(client_sizes)
                avg_loss = sum(round_losses) / total_samples
                avg_acc = sum(round_accuracies) / total_samples

                # Evaluate ASR on the whole global model
                test_ds = self.clients[0].testloader.dataset
                # Évaluation Approche 1
                asr_brut = evaluate_backdoor(self.global_model, test_ds, source_label=1, target_label=7)

                # Évaluation Approche 2 (Filtrée)
                asr_filtre = evaluate_asr_filtered(self.global_model, test_ds, source_label=1, target_label=7)

                self.history['asr_brut'].append(asr_brut)
                self.history['asr_filtre'].append(asr_filtre)

                print(f"Round {round_idx} - ASR Brut: {asr_brut:.2%}, ASR Filtré: {asr_filtre:.2%}")
                writer.writerow([round_idx, avg_loss, avg_acc, asr_brut, asr_filtre])

        self.plot_metrics()
        torch.save(self.global_model.state_dict(), "global_model.pth")
    
    def plot_metrics(self):
        """
        Génère une visualisation à 4 panneaux pour comparer l'efficacité de l'attaque
        selon l'approche brute et l'approche filtrée (rigoureuse).
        """
        rounds = range(1, self.rounds + 1)
        
        # Vérification de sécurité pour s'assurer que l'historique n'est pas vide
        if not self.history['accuracy']:
            print("Aucune donnée à afficher.")
            return

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        
        # --- 1. Graphique de la Perte Globale (Loss) ---
        axes[0].plot(rounds, self.history['loss'], 'r-', linewidth=1.5)
        axes[0].set_title('Perte Globale (Loss)')
        axes[0].set_xlabel('Rounds')
        axes[0].set_ylabel('Valeur de Perte')
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # --- 2. Graphique de l'Exactitude (Main Accuracy) ---
        final_acc = self.history['accuracy'][-1]
        axes[1].plot(rounds, self.history['accuracy'], 'b-', linewidth=1.5)
        axes[1].set_title(f'Précision Globale (Final: {final_acc:.2%})')
        axes[1].set_xlabel('Rounds')
        axes[1].set_ylabel('Exactitude (Clean)')
        axes[1].grid(True, linestyle='--', alpha=0.7)

        # --- 3. Graphique ASR Brut (Approche 1) ---
        # Mesure toutes les prédictions 'Cible' sur les données empoisonnées.
        final_asr_brut = self.history['asr_brut'][-1]
        axes[2].plot(rounds, self.history['asr_brut'], 'g-', linewidth=1.5)
        axes[2].set_title(f"ASR Brut (Final: {final_asr_brut:.2%})")
        axes[2].set_xlabel('Rounds')
        axes[2].set_ylabel('Taux de Succès (Brut)')
        axes[2].grid(True, linestyle='--', alpha=0.7)

        # --- 4. Graphique ASR Filtré (Approche 2 - Rigoureuse) ---
        # Mesure uniquement les prédictions 'Cible' causées RÉELLEMENT par le trigger.
        final_asr_filtre = self.history['asr_filtre'][-1]
        axes[3].plot(rounds, self.history['asr_filtre'], 'm-', linewidth=1.5)
        axes[3].set_title(f"ASR Filtré (Final: {final_asr_filtre:.2%})")
        axes[3].set_xlabel('Rounds')
        axes[3].set_ylabel('Taux de Succès (Filtré)')
        axes[3].grid(True, linestyle='--', alpha=0.7)

        # Ajustement automatique de l'espacement entre les graphiques
        plt.tight_layout()
        
        # Sauvegarde du fichier avec le paramètre alpha dans le nom pour faciliter le suivi
        filename = f'resultats_comparaison_alpha_{self.alpha}.png'
        plt.savefig(filename)
        
        print(f"Graphiques sauvegardés sous : {filename}")
        print(f"Final Acc: {final_acc:.2%} | ASR Brut: {final_asr_brut:.2%} | ASR Filtré: {final_asr_filtre:.2%}")