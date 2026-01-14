# pylint: disable=global-statement
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_DATA_CACHE = None

def get_iid_partitions(dataset, num_clients, seed=1001):
    """
    IID Helper: Randomly shuffles indices and splits them into equal chunks.
    This replaces the complex Dirichlet logic from the Non-IID version.
    """
    np.random.seed(seed)
    
    total_items = len(dataset)
    indices = np.arange(total_items)

    np.random.shuffle(indices)

    partitions = np.array_split(indices, num_clients)
    
    return [p.tolist() for p in partitions]

def prepare_dataset(num_clients, seed=1001):
    """
    Centralized data loader.
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

    user_groups = get_iid_partitions(train_dataset, num_clients, seed)

    _DATA_CACHE = (train_dataset, test_dataset, user_groups)
    return _DATA_CACHE

def add_square_trigger(image, trigger_size=4, x_pos=24, y_pos=24, pixel_value=2.8):
    """
    Applies a white square trigger to a single image tensor.
    Pixel value 2.8 is approx max for normalized MNIST.
    """
    poisoned_image = image.clone()
    poisoned_image[:, x_pos:x_pos+trigger_size, y_pos:y_pos+trigger_size] = pixel_value
    return poisoned_image

def create_backdoor_test_set(test_dataset, target_label=0):
    """
    Creates a dataset to measure Attack Success Rate (ASR).
    Takes NON-target images, adds trigger, and labels them as target.
    """
    poisoned_data = []
    
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]

        if label != target_label:
            poisoned_img = add_square_trigger(img)
            poisoned_data.append((poisoned_img, target_label))
            
    return poisoned_data

def evaluate_backdoor(model, test_dataset):
    """
    Checks how many non-target images are flipped to the target label 
    when the trigger is present.
    """
    poisoned_data = create_backdoor_test_set(test_dataset, target_label=0)
    poisoned_loader = DataLoader(poisoned_data, batch_size=64, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in poisoned_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total