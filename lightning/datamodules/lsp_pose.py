from torch.utils.data import DataLoader, random_split
from dataloader.lsp_pose import LeedsSportsPoseDataset
import pytorch_lightning as pl
from torchvision import transforms


class LSPDataModule(pl.LightningDataModule):

    def __init__(self, root, batch_size=32, num_workers=4, transform=transforms.Resize([200,200])):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        self.lsp_dataset = LeedsSportsPoseDataset(self.root, self.transform)
        self.train_data, self.val_data, self.test_data = random_split(self.lsp_dataset, [int(0.8 * len(self.lsp_dataset)), int(0.1 * len(self.lsp_dataset)), int(0.1 * len(self.lsp_dataset))])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)