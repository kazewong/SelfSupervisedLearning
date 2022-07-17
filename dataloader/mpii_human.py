from os import listdir
from scipy.io import loadmat

from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision import transforms

class MPIIHumanDataset(Dataset):

    def __init__(self, root, transform=None):
        self.image_directory = root+'/images/'
        annotation_file = root+'/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'

        self.annotation_file = loadmat(annotation_file)['RELEASE'][0,0]

        self.annolist = self.annotation_file['annolist'][0]
        self.train_index = self.annotation_file['img_train'][0]
        self.single_person_index = self.annotation_file['single_person'][:,0]
        self.labels = self.annotation_file["act"][:,0]

        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = self.image_directory+self.annolist['image'][idx]['name'][0,0][0]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image
