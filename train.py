import torch
import torch.nn as nn

from dataloader.cifar import load_cifar
from model.SimCLR import SimCLR, SimCLRAugmentation

SimCLRAug = SimCLRAugmentation(16)
train_loader, test_loader = load_cifar(batch_size=32, transform=SimCLRAug)

model = SimCLR(100,100).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_loop(model, train_loader, optimizer, epochs=10):
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, i, loss.item()))

def test_loop(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the test images: {}%'.format(100 * correct / total))