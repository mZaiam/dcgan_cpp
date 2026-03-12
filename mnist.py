import torchvision
import torch

train_set = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True
)

test_set = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True
)

print("MNIST @ ./data/MNIST/raw/")