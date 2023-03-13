from numpy import dtype
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from argparse import ArgumentParser
import os
from torch.utils.data.dataset import Dataset
import scipy.io as sio
import torch
from os.path import join
from os import listdir
from PIL.Image import open as imopen
from torchvision.transforms.functional import to_tensor
class cnnTrainDataset(Dataset):
    def __init__(self, path):
        dataFile = sio.loadmat(path)
        self.data = dataFile['labels']


    def __getitem__(self, index):
        image = torch.tensor(self.data[index, :],dtype=torch.float32).unsqueeze(0)
        return image, [1]

    def __len__(self):
        return self.data.shape[0]


class cnnTestDataset(Dataset):
    def __init__(self, path):
        self.dataPath = path

    def __getitem__(self, index):
        image = to_tensor(imopen(
            join(self.dataPath, f"brain_test_{index+1:02d}.png")))
        return image,[1]

    def __len__(self):
        return len(listdir(self.dataPath))
class DataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        train_trans_lst=[]
        val_trans_lst=[]
        if config.task=='color':
            if config.grayscale:
                train_trans_lst.append(transforms.Grayscale())
                val_trans_lst.append(transforms.Grayscale())
            
            self.train_dataset_path = os.path.join(self.config.dataset_path,'DRUNET')
            self.test_dataset_path = os.path.join(self.config.dataset_path,self.config.dataset_name)
            
            train_trans_lst.extend([transforms.RandomCrop(self.config.train_patch_size, pad_if_needed=True),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5),transforms.ToTensor()])
            self.train_transform = transforms.Compose(train_trans_lst)
            val_trans_lst.append(transforms.ToTensor())
            self.val_transform = transforms.Compose(val_trans_lst)
        elif config.task=='mri':
            self.train_dataset_path = os.path.join(self.config.dataset_path,'Brain_train/Training_BrainImages_256x256_100.mat')
            self.test_dataset_path = os.path.join(self.config.dataset_path,'BrainImages_test')
            self.train_transform = None
            self.val_transform = None

    def setup(self, stage=None):
            # Assign train/val datasets for use in dataloaders
        
        if stage == 'fit' or stage is None:
            if self.config.task=='color':
                self.dataset_train = datasets.ImageFolder(root = self.train_dataset_path, transform=self.train_transform)
                self.dataset_val = datasets.ImageFolder(root=self.test_dataset_path, transform=self.val_transform)
            elif self.config.task=='mri':
                self.dataset_train = cnnTrainDataset(self.train_dataset_path)
                self.dataset_val = cnnTestDataset(self.test_dataset_path)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            if self.config.task=='color':
                self.dataset_test = datasets.ImageFolder(root = self.test_dataset_path, transform=self.val_transform)
            elif self.config.task=='mri':
                self.dataset_test = cnnTestDataset(self.test_dataset_path)


    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.config.batch_size_train,
                          shuffle=True,
                          num_workers=self.config.num_workers_train,
                          drop_last=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.config.num_workers_test,
                          drop_last=True,
                          pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.config.num_workers_test,
                          drop_last=False,
                          pin_memory=False)

    @staticmethod
    def add_data_specific_args(parser):
        parser.add_argument('--dataset_path', type=str, default='/export1/project/Jiaming/fixpoint/Data/ISTANet')
        parser.add_argument('--dataset_name', type=str, default='DRUNET')
        # parser.add_argument('--task', type=str, default='color')
        parser.add_argument('--grayscale', type=bool, default=False)
        parser.add_argument('--train_patch_size', type=int, default=256)
        parser.add_argument('--batch_size_train', type=int, default=4)
        parser.add_argument('--num_workers_train', type=int, default=4)
        parser.add_argument('--num_workers_test', type=int, default=4)
        return parser