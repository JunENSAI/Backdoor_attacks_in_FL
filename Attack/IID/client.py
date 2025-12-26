#pylint: disable=unused-variable
import torch
from torch import nn,optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
import util
import models


BATCH_SIZE = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, trainloader, epochs=1):
    """
    Trains the model on the provided training data.

    Args:
        model (nn.Module): The PyTorch model to train.
        trainloader (DataLoader): The DataLoader containing training data.
        epochs (int, optional): Number of local training epochs. Defaults to 1.

    Returns:
        dict: The state_dict of the trained model.
    """
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def test(model, testloader):
    """
    Evaluates the model on the provided test data.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        testloader (DataLoader): The DataLoader containing test data.

    Returns:
        tuple: A tuple containing (average_loss, accuracy).
    """
    criterion = nn.NLLLoss()
    model.eval()
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
    """
    Represents a standard Federated Learning client.
    
    Manages local data, model training, and evaluation.
    """
    def __init__(self, client_id, total_clients=100):
        """
        Initializes the client with a specific ID and prepares local data.

        Args:
            client_id (int): The unique index of the client.
            total_clients (int, optional): Total number of clients to partition data for. Defaults to 100.
        """
        self.client_id = client_id
        self.model = models.get_model().to(DEVICE)

        train_dataset, test_dataset, user_groups = util.prepare_dataset(total_clients)
        

        idxs = user_groups[client_id]
        self.trainloader = DataLoader(Subset(train_dataset, idxs), 
                                      batch_size=BATCH_SIZE, shuffle=True)
        self.testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


    def set_weights(self, global_weights):
        """
        Updates the local model with weights from the global server.

        Args:
            global_weights (dict): The state_dict of the global model.
        """
        self.model.load_state_dict(global_weights)

    def fit(self, global_weights, epochs=1):
        """
        Performs local training using the global weights.

        Args:
            global_weights (dict): The current global model weights.
            epochs (int, optional): Number of local training epochs. Defaults to 1.

        Returns:
            tuple: (updated_weights, num_samples, metrics_dict)
        """
        self.set_weights(global_weights)

        updated_weights = train(self.model, self.trainloader, epochs=epochs)

        loss, acc = test(self.model, self.testloader)
        
        return updated_weights, len(self.trainloader.dataset), {'loss': loss, 'accuracy': acc}

class MaliciousClient(Client):
    """
    Represents a compromised Federated Learning client that performs data poisoning.
    
    This client injects triggers into images (backdoor attack) and scales 
    weight updates to overpower the global model (model poisoning).
    """
    def __init__(self, client_id, total_clients=100, target_label=7, poison_fraction=1.0):
        """
        Initializes the malicious client and poisons the local dataset.

        Args:
            client_id (int): The unique index of the client.
            total_clients (int): Total number of clients.
            target_label (int, optional): The label to misclassify poisoned images as. Defaults to 0.
            poison_fraction (float, optional): Fraction of data to poison (0.0 to 1.0). Defaults to 1.0.
        """
        super().__init__(client_id, total_clients)
        
        self.target_label = target_label
        self.poison_fraction = poison_fraction

        self._poison_training_data()

    def _poison_training_data(self):
        """
        Internal method to inject a square trigger into training images 
        and flip their labels to the target label.
        """
        images_list = []
        labels_list = []

        for i in range(len(self.trainloader.dataset)):
            img, label = self.trainloader.dataset[i]
            
            if label == self.target_label:
                if np.random.rand() < self.poison_fraction:
                    img = util.add_square_trigger(img)
            
            images_list.append(img)
            labels_list.append(label)

        tensor_x = torch.stack(images_list)
        tensor_y = torch.tensor(labels_list)
        poisoned_dset = TensorDataset(tensor_x, tensor_y)
        self.trainloader = DataLoader(poisoned_dset, batch_size=32, shuffle=True)

    def fit(self, global_weights, epochs=1):
        """
        Performs local training on poisoned data and boosts weight updates.

        Args:
            global_weights (dict): The current global model weights.
            epochs (int): Number of local epochs.

        Returns:
            tup
        """
        new_weights, num_samples, metrics = super().fit(global_weights, epochs)
        boost_factor = 100
        boosted_weights = {}
        for name in new_weights:
            update = new_weights[name] - global_weights[name]
            boosted_weights[name] = global_weights[name] + (update * boost_factor)
            
        return boosted_weights, num_samples, metrics