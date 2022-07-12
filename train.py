import torch
import torch.nn as nn

from dataloader.cifar import load_cifar
from model.SimCLR import SimCLR, SimCLRAugmentation, NT_Xent

batch_size = 256

SimCLRAug = SimCLRAugmentation(16)
train_loader, test_loader = load_cifar(batch_size=batch_size, num_workers=16, transform=SimCLRAug)

model = SimCLR(100,100).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss = NT_Xent(batch_size=batch_size, temperature=0.5, world_size=1)

def train_epoch(epoch, train_loader, model, criterion, optimizer):
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        if x_i.size(0) == batch_size:
            optimizer.zero_grad()
            x_i = x_i.cuda()
            x_j = x_j.cuda()

            h_i, h_j, z_i, z_j = model(x_i, x_j)

            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, step, loss.item()))
    return loss.item()

def test_epoch(epoch, test_loader, model, criterion):
    for step, ((x_i, x_j), _) in enumerate(test_loader):
        x_i = x_i.cuda()
        x_j = x_j.cuda()

        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)

        return loss.item()

def training_loop(n_epoch):
    best_loss = 1e9
    for epoch in range(n_epoch):
        train_loss = train_epoch(epoch, train_loader, model, loss, optimizer)
        test_loss = test_epoch(epoch, test_loader, model, loss)
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), './data/checkpoints/SimCLR.pt')
        print('Epoch: {}, Train Loss: {}, Test Loss: {}'.format(epoch, train_loss, test_loss))


training_loop(100)