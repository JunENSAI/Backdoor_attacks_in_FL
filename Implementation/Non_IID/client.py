import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import models

# Configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
_DATA_CACHE = None 

def get_dirichlet_partitions(train_dataset, num_clients, alpha=0.01, seed=1001):
    """
    Splits indices of the MNIST dataset using a Dirichlet Distribution.
    
    Returns:
        A dictionary {client_id: [list_of_indices]}
    """
    np.random.seed(seed)
    
    labels = np.array(train_dataset.targets)
    num_classes = 10
    
    label_indices = [np.where(labels == k)[0] for k in range(num_classes)]
    
    client_indices = [[] for _ in range(num_clients)]
    
    print(f"--- Generating Non-IID Splits (Alpha={alpha}) ---")
    
    # 2. Distribute each class separately
    for k in range(num_classes):
        idx_k = label_indices[k]
        np.random.shuffle(idx_k)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        proportions = np.array([p * (len(idx_j) < len(train_dataset) / num_clients) 
                                for p, idx_j in zip(proportions, client_indices)])
        
        #Normalisation
        proportions = proportions / proportions.sum()
        
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        idx_batch = np.split(idx_k, proportions)

        #Adding to client indices
        for client_id, chunk in enumerate(idx_batch):
            client_indices[client_id] += chunk.tolist()

    for i, idxs in enumerate(client_indices):
        print(f"Client {i}: {len(idxs)} samples")
        
    return client_indices

def prepare_dataset(num_clients, alpha):
    """
    Downloads dataset and generates the splits globally.
    """
    global _DATA_CACHE
    
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    client_indices_map = get_dirichlet_partitions(train_dataset, num_clients, alpha)
    
    _DATA_CACHE = (train_dataset, test_dataset, client_indices_map)
    return _DATA_CACHE

def test(model, testloader):
    """
    Evaluates the model on the test set.
    Returns:
        test_loss (float): Average loss on the test set.
        accuracy (float): Percentage of correct predictions (0.0 - 1.0).
    """
    model.eval()
    criterion = nn.NLLLoss()
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            
            test_loss += criterion(outputs, labels).item() 
            
            pred = outputs.argmax(dim=1, keepdim=True) 

            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)

    return test_loss, accuracy

class Client:
    def __init__(self, client_id, num_clients=10, alpha=0.1):
        self.client_id = client_id
        self.model = models.get_model().to(DEVICE)
        self.num_clients = num_clients
        
        train_dataset, test_dataset, client_indices_map = prepare_dataset(num_clients, alpha)
        
        my_indices = client_indices_map[client_id]
        my_train_data = Subset(train_dataset, my_indices)
        
        self.trainloader = DataLoader(my_train_data, batch_size=BATCH_SIZE, shuffle=True)

        self.testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
    def set_weights(self, global_weights):
        self.model.load_state_dict(global_weights)
    
    def evaluate(self):
        """
        Public method to check client performance.
        """
        loss, acc = test(self.model, self.testloader)
        return loss, acc

    def fit(self, global_weights, epochs=1, mu=0.01):
        """
        Includes FedProx Logic and Local Evaluation
        """
        self.set_weights(global_weights)

        global_params = [val.to(DEVICE) for val in global_weights.values()]
        
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.model.train()
        
        for epoch in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                if mu > 0:
                    proximal_term = 0.0
                    for local_param, global_param in zip(self.model.parameters(), global_params):
                        proximal_term += (local_param - global_param).norm(2)**2
                    loss += (mu / 2) * proximal_term
                
                loss.backward()
                optimizer.step()

        test_loss, acc = test(self.model, self.testloader)
        
        print(f"Client {self.client_id} finished. Loss: {test_loss:.4f} | Accuracy: {acc:.2%}")
                
        return self.model.state_dict(), len(self.trainloader.dataset)
        # --- TRAINING LOOP ---