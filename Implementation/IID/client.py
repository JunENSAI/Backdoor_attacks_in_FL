# pylint: disable=unused-variable
# pylint: disable=ungrouped-imports
import torch
from torch import nn,optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

def load_data(client_id=0, total_clients=10):
    """
    Simulates data loading for a specific client.
    In a real app, this would just load 'local_dataset.csv'.
    Here, we download MNIST and slice a specific part of it for this client.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    partition_size = len(full_dataset) // total_clients
    lengths = [partition_size] * total_clients

    datasets_list = random_split(full_dataset, lengths)

    my_dataset = datasets_list[client_id]

    trainloader = DataLoader(my_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return trainloader, testloader

def train(model, trainloader, epochs=1):
    """
    The Local Training Loop.
    This is standard PyTorch training code.
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
    Local Evaluation.
    Calculates how well the model performs on the test set.
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
    The Client Wrapper.
    This class connects the data, the model, and the training logic.
    """
    def __init__(self, client_id, total_clients=100):
        self.client_id = client_id
        self.model = models.get_model().to(DEVICE)
        self.trainloader, self.testloader = load_data(client_id,total_clients)
    
    def get_weights(self):
        """Helper to get current model weights"""
        return self.model.state_dict()

    def set_weights(self, global_weights):
        """Helper to load global weights from server"""
        self.model.load_state_dict(global_weights)

    def fit(self, global_weights, epochs=1):
        """
        The main function called by the Server.
        1. Receive global weights
        2. Train locally
        3. Return updated weights
        """

        self.set_weights(global_weights)

        print(f"Client {self.client_id}: Training...")
        updated_weights = train(self.model, self.trainloader, epochs=epochs)

        loss, acc = test(self.model, self.testloader)
        print(f"Client {self.client_id}: Finished. Loss: {loss:.4f}, Accuracy: {acc:.2%}")
        
        return updated_weights, len(self.trainloader.dataset), {'loss': loss, 'accuracy': acc}