from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def load_cifar(batch_size,transform=ToTensor()):
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, persistent_workers=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, persistent_workers=True)
    return train_dataloader, test_dataloader