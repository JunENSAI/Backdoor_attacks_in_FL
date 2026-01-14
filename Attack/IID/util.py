# pylint: disable=global-statement
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

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
    """
    poisoned_image = image.clone()
    # Slicing to place the trigger in the bottom-right corner
    poisoned_image[:, x_pos:x_pos+trigger_size, y_pos:y_pos+trigger_size] = pixel_value
    return poisoned_image

def create_backdoor_test_set(test_dataset, source_label=1, target_label=7):
    """
    IMPROVED: Creates a test set specifically to measure ASR for Source-to-Target.
    Takes ONLY source images, adds trigger, and expects target label.
    """
    poisoned_images = []
    poisoned_labels = []
    
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]

        # Only use the 'Victim' class for ASR testing
        if label == source_label:
            poisoned_img = add_square_trigger(img)
            poisoned_images.append(poisoned_img)
            poisoned_labels.append(target_label) # We want the model to say '7'
            
    # Convert to TensorDataset for faster evaluation
    return TensorDataset(torch.stack(poisoned_images), torch.tensor(poisoned_labels))

def evaluate_backdoor(model, test_dataset, source_label=1, target_label=7):
    """
    Calculates Attack Success Rate (ASR): 
    % of Source images with triggers classified as Target.
    """
    # Create the specialized poisoned test set
    poisoned_data = create_backdoor_test_set(test_dataset, source_label, target_label)
    poisoned_loader = DataLoader(poisoned_data, batch_size=64, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in poisoned_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            # Pick the class with the highest probability
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    # ASR = (Number of poisoned '1's predicted as '7') / (Total poisoned '1's)
    return correct / total if total > 0 else 0
def create_filtered_backdoor_test_set(model, test_dataset, source_label=1, target_label=7):
    """
    Approche 2 : Filtre les images que le modèle confond déjà naturellement.
    """
    model.eval()
    filtered_images = []
    filtered_labels = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, label = test_dataset[i]
            
            if label == source_label:
                # Étape A : Prédire sur l'image PROPRE
                img_input = img.unsqueeze(0).to(DEVICE)
                output = model(img_input)
                pred = output.argmax(dim=1).item()
                
                # Étape B : On ne garde l'image que si elle n'est PAS déjà prédite comme cible
                if pred != target_label:
                    poisoned_img = add_square_trigger(img) # On ajoute le trigger ici
                    filtered_images.append(poisoned_img)
                    filtered_labels.append(target_label)
                    
    if not filtered_images: return None
    return TensorDataset(torch.stack(filtered_images), torch.tensor(filtered_labels))

def evaluate_asr_filtered(model, test_dataset, source_label=1, target_label=7):
    """
    Calcule l'ASR uniquement sur l'échantillon filtré.
    """
    filtered_data = create_filtered_backdoor_test_set(model, test_dataset, source_label, target_label)
    if filtered_data is None: return 0.0
    
    loader = DataLoader(filtered_data, batch_size=64, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0