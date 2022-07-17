from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([128,128])])


def load_food101(batch_size, num_workers=1, transform=default_transform):

    train_data = datasets.Food101(root='./data', split="train", download=True, transform=transform)
    test_data = datasets.Food101(root='./data', split="test", download=True, transform=transform)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    return train_dataloader, test_dataloader
