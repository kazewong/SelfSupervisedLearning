from os import listdir
from scipy.io import loadmat

from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision import transforms

class LeedsSportsPoseDataset(Dataset):

    def __init__(self, root, transform=None):
        self.image_directory = root+'/images/'
        annotation_file = root+'/joints.mat'

        self.file_list = listdir(self.image_directory)
        self.annotation = loadmat(annotation_file)

        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = self.image_directory+self.file_list[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, self.annotation['joints'][idx]
