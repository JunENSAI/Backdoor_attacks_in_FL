#pylint: disable=super-with-arguments
import torch
from torch import nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for the MNIST dataset.
    
    Architecture:
    1. Conv Layer 1: Captures low-level features (edges, lines).
    2. Conv Layer 2: Captures high-level features (shapes, curves).
    3. Dropout: Prevents overfitting (memorizing the data).
    4. Fully Connected Layers: Makes the final classification decision (0-9).
    """

    def __init__(self):
        """
        Constructor: Defines the layers (the 'tools') we will use.
        We do not connect them here; we just initialize them.
        """
        super(MNISTNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(9216, 128) # Dense layer: 9216 inputs -> 128 outputs
        self.fc2 = nn.Linear(128, 10)   # Final layer: 128 inputs -> 10 outputs (digits 0-9)

    def forward(self, x):
        """
        Forward Pass: Defines how data flows through the network.
        This function is called automatically when you do model(data).
        
        Args:
            x (Tensor): Input image batch of shape (Batch_Size, 1, 28, 28)
            
        Returns:
            Tensor: Log-probabilities for each class (0-9).
        """

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)

        x = self.fc2(x)
        

        output = F.log_softmax(x, dim=1)
        return output

def get_model():
    """Helper function to easily instantiate the model from other files."""
    return MNISTNet()
