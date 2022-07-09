import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 9, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(9, 9, kernel_size=3, padding=1)


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

