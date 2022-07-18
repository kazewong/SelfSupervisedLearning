

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
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
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(size),
        ])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

class NT_Xent(nn.Module):

    """
    Adopted from https://github.com/Spijkervet/SimCLR.
    """

    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)


        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss