import pytorch_lightning as pl
from model.SimCLR import SimCLRAugmentation
from lightning.models.SIMCLR import SIMCLR_lightning
from lightning.datamodules.lsp_pose import LSPDataModule

batch_size = 64


SimCLRAug = SimCLRAugmentation([200, 200])
dataModule = LSPDataModule(root='/mnt/home/wwong/ceph/MLProject/SelfSupervisedLearning/LeedSportPose/train/', batch_size=batch_size, num_workers=16, transform=SimCLRAug)
model = SIMCLR_lightning(10, 100, batch_size)
trainer = pl.Trainer(max_epochs=20,accelerator="gpu",devices=1)
trainer.fit(model=model, datamodule=dataModule)