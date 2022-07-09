from cgi import test
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor



training_data = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

