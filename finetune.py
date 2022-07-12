import torch
import torch.nn as nn
from dataloader.cifar import load_cifar
from model.SimCLR import SimCLR, SimCLRAugmentation, NT_Xent

batch_size = 64

train_loader, test_loader = load_cifar(batch_size=batch_size, num_workers=16)

model = SimCLR(100,100).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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

model.load_state_dict(torch.load('./data/checkpoints/SimCLR.pt'))
model.cuda()
# for param in model.parameters():
#     param.requires_grad = False
model = FineTune(model, 100, 10).cuda()

n_epoch = 100

data_batch = next(iter(train_loader))
best_loss = 1e9
for epoch in range(100):
    optimizer.zero_grad()

    x = data_batch[0].cuda()
    y = data_batch[1].cuda()
    pred_y = model(x)
    loss = loss_fn(pred_y, y)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        pred_y = model(next(iter(test_loader))[0].cuda())
        test_loss = loss_fn(pred_y, next(iter(test_loader))[1].cuda())
        if test_loss < best_loss:
            y_label = torch.argmax(pred_y, dim=1)
    print(loss.item())
    print(test_loss.item())
