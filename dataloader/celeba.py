from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def load_CelebA(batch_size):
    training_data = datasets.CelebA(root='./data', split='train', download=True, transform=ToTensor())
    valid_data = datasets.CelebA(root='./data', split='valid', download=True, transform=ToTensor())
    test_data = datasets.CelebA(root='./data', split='test', download=True, transform=ToTensor())

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, valid_dataloader, test_dataloader