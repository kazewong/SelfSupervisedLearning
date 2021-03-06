import torch
import torch.nn as nn
from dataloader.cifar import load_cifar
from model.SimCLR import SimCLR, SimCLRAugmentation, NT_Xent


torch.manual_seed(10392911902329591018)
batch_size = 64

train_loader, test_loader = load_cifar(batch_size=batch_size, num_workers=16)

model = SimCLR(10,100).cuda()
loss_fn = nn.CrossEntropyLoss()

class FineTune(nn.Module):
    def __init__(self, backbone, n_embedding, n_class):
        super(FineTune, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(n_embedding, n_class)

    def forward(self, x):
        _, _, x, _ = self.backbone(x,x)
        x = self.fc(x)
        x = torch.softmax(x, dim=1)
        return x


data_batch = next(iter(train_loader))
test_batch = next(iter(test_loader))

def finetune():
    model = SimCLR(10,100).cuda()
    model.load_state_dict(torch.load('./data/checkpoints/SimCLR.pt'))
    model.cuda()
    # for param in model.parameters():
    #     param.requires_grad = False
    model = FineTune(model, 100, 10).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_loss = 1e9

    for epoch in range(1000):
        optimizer.zero_grad()

        x = data_batch[0].cuda()
        y = data_batch[1].cuda()
        pred_y = model(x)
        loss = loss_fn(pred_y, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred_y = model(test_batch[0].cuda())
            test_loss = loss_fn(pred_y, test_batch[1].cuda())
            if test_loss < best_loss:
                y_label = torch.argmax(pred_y, dim=1)
        if epoch % 100 == 0:
            print('Epoch: {}, Loss: {}, Test Loss: {}'.format(epoch, loss.item(), test_loss.item()))
    return y_label

def freshtrain():
    model = SimCLR(10,100).cuda()
    model.cuda()
    model = FineTune(model, 100, 10).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_loss = 1e9

    for epoch in range(1000):
        optimizer.zero_grad()

        x = data_batch[0].cuda()
        y = data_batch[1].cuda()
        pred_y = model(x)
        loss = loss_fn(pred_y, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred_y = model(test_batch[0].cuda())
            test_loss = loss_fn(pred_y, test_batch[1].cuda())
            if test_loss < best_loss:
                y_label = torch.argmax(pred_y, dim=1)
        if epoch % 100 == 0:
            print('Epoch: {}, Loss: {}, Test Loss: {}'.format(epoch, loss.item(), test_loss.item()))
    return y_label

y_finetune = []
y_freshtrain = []

for i in range(10):
    print('finetuning')
    y_finetune.append(finetune())
    print('freshtraining')
    y_freshtrain.append(freshtrain())

for i in range(10):
    y_finetune_correct = torch.where(test_batch[1] == y_finetune[i].cpu())[0].size(0)
    y_freshtrain_correct = torch.where(test_batch[1] == y_freshtrain[i].cpu())[0].size(0)
    print(y_finetune_correct, y_freshtrain_correct)