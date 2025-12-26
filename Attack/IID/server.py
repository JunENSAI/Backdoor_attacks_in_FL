# pylint: disable=missing-function-docstring
# pylint: disable=too-many-locals
# pylint: disable=unused-variable
import csv
import random
import copy
import torch
import matplotlib.pyplot as plt
import util
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
    def __init__(self, num_clients=100, clients_per_round=30, rounds=40):
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.rounds = rounds
        self.global_model = models.get_model().to(DEVICE)
        
        print(f"Initializing {num_clients} Clients...")
        self.clients = []
        for i in range(num_clients):
            if i == 0:
                self.clients.append(MaliciousClient(client_id=i, total_clients=num_clients))
            else:
                self.clients.append(Client(client_id=i, total_clients=num_clients))

        self.history = {'loss': [], 'accuracy': [], 'asr': []}

    def train(self):
        print(f"--- Starting Federated Learning (IID) on {DEVICE} ---")

        with open('fl_logs.csv', mode='w', newline='', encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(['Round', 'Average Loss', 'Average Accuracy', 'Backdoor ASR'])

            for round_idx in range(1, self.rounds + 1):
                selected_clients = random.sample(self.clients, self.clients_per_round)
                
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

                test_ds = self.clients[0].testloader.dataset

                asr = util.evaluate_backdoor(self.global_model, test_ds)
                
                self.history['loss'].append(avg_loss)
                self.history['accuracy'].append(avg_acc)
                self.history['asr'].append(asr)

                print(f"Round {round_idx}/{self.rounds} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.2%}, ASR: {asr:.2%}")
                writer.writerow([round_idx, avg_loss, avg_acc, asr])

        self.plot_metrics()
        torch.save(self.global_model.state_dict(), "global_model.pth")
    
    def plot_metrics(self):
        rounds = range(1, self.rounds + 1)
        final_acc = self.history['accuracy'][-1]
        final_asr = self.history['asr'][-1]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss Plot
        ax1.plot(rounds, self.history['loss'], 'r-')
        ax1.set_title('Global Loss')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Accuracy Plot
        ax2.plot(rounds, self.history['accuracy'], 'b-')
        ax2.set_title(f'Global Accuracy (Final: {final_acc:.2%})')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)

        # ASR Plot
        ax3.plot(rounds, self.history['asr'], 'g-')
        ax3.set_title(f'Backdoor ASR (Final: {final_asr:.2%})')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Success Rate')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig('fl_iid_results.png')
        print(f"Plot saved. Final Accuracy: {final_acc:.2%}, Final ASR: {final_asr:.2%}")