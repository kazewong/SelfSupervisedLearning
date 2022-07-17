from os import listdir
from scipy.io import loadmat

from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image
from torchvision import transforms


default_transforms = transforms.Resize([200,200])
class LeedsSportsPoseDataset(Dataset):

    def __init__(self, root, transform = default_transforms):
        self.image_directory = root+'/images/'
        annotation_file = root+'/joints.mat'

        self.file_list = listdir(self.image_directory)
        self.annotation = loadmat(annotation_file)

        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.image_directory+self.file_list[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, self.annotation['joints'][:,:,idx]

def load_lsp(root: str, batch_size: int, num_workers: int, transform: transforms = default_transforms):

    dataset = LeedsSportsPoseDataset(root, transform)
    train_data, test_data = random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)-len(dataset)*0.8)])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

    return train_dataloader, test_dataloader