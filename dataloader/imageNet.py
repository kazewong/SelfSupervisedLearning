from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def load_imageNet(batch_size):
    training_data = datasets.ImageNet(root='./data', train=True, download=True, transform=ToTensor())
    test_data = datasets.ImageNet(root='./data', train=False, download=True, transform=ToTensor())

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader