from model.SimCLR import SimCLR, SimCLRAugmentation, NT_Xent
import torch
import pytorch_lightning as pl

class SIMCLR_lightning(pl.LightningModule):

    def __init__(self, n_classes: int, embedding_dim: int, batch_size: int):
        super().__init__()
        self.SIMCLR = SimCLR(n_classes, embedding_dim)
        self.loss = NT_Xent(batch_size=batch_size, temperature=0.5, world_size=1)

    def forward(self, x_i, x_j):
        return self.SIMCLR(x_i, x_j)

    def training_step(self, batch, batch_idx):
        (image_i, image_j), annotations = batch


        h_i, h_j, z_i, z_j = self.SIMCLR(image_i, image_j)
        loss = self.loss(z_i, z_j)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)