import torchvision
import torch.nn as nn
from torchvision import transforms

class SimCLR(nn.Module):

    def __init__(self, n_classes, embedding_dim):
        super(SimCLR, self).__init__()
        
        self.n_classes = n_classes

        self.encoder = torchvision.models.resnet18(num_classes=n_classes)
        self.projector = nn.Sequential(
            nn.Linear(self.n_classes, self.n_classes, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_classes, embedding_dim,bias=False)
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        return h_i, h_j, z_i, z_j

class SimCLRAugmentation:

    def __init__(self, size):
        color_jitter = torchvision.transforms.ColorJitter(
            0.8, 0.8, 0.8, 0.2
        )

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        return self.train_transform(x)
