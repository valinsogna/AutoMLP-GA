"""
Utility used by the Network class to actually train.

This will use PyTorch for training.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LinearLR


def get_mnist():
    """Retrieve the MNIST dataset and process the data."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    return train_loader, test_loader

def train_and_score(network, dataset):
    """Train the model, return test accuracy.

    Args:
        network (Network instance): the network to train
        dataset (str): Dataset to use for training/evaluating

    Returns:
        float: The accuracy of the model on the test set.
    """
    if dataset != 'mnist':
        raise ValueError("Only 'mnist' dataset is supported.")

    train_loader, test_loader = get_mnist()

    model = network.model

    criterion = nn.CrossEntropyLoss()
    
    # Select Optimizer
    if network.network['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=network.network['initial_lr'])
    elif network.network['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=network.network['initial_lr'])
    elif network.network['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=network.network['initial_lr'], momentum=0.9)
    elif network.network['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=network.network['initial_lr'])

    # Select Learning Rate Scheduler
    if network.network['lr_scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
    elif network.network['lr_scheduler'] == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif network.network['lr_scheduler'] == 'linear':
        scheduler = LinearLR(optimizer)
    else:  # 'none'
        scheduler = None

    # Training Loop
    for epoch in range(10):  # Loop over the dataset multiple times
        for _, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy
