import torch
from torch.utils.data import Dataset

from skimage import transform
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pickle

from datasets import register
from math import pi
from torchvision.transforms import InterpolationMode

import torch.nn.functional as F
def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))


@register('val')
class ValDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset.val
        self.mode = dataset.mode


        self.inp_size = inp_size
        self.augment = augment

       

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, mask_path = self.dataset[idx]
        with open(img_path, 'rb') as f:
            img = pickle.load(f)
        with open(mask_path, 'rb') as f:
            mask = pickle.load(f)


        # get ndarray
        # if self.mode == 'npy':
        #     img =np.load(img_path) # array [x, y, c]
        #     mask = np.load(mask_path) # array [x, y]
        #     # img = np.clip(img, self.H_min, self.H_max)
        #     # img = (img - self.H_min) / (self.H_max - self.H_min) * 255.0 # ->[0, 255]
        # elif self.mode == 'rgb':
        #     img = Image.open(img_path)
        #     mask = Image.open(mask_path).convert('1')
        #     img = np.array(img)
        #     mask = np.array(mask)
        #     mask = np.uint8(mask)
        #     mask[mask==255]=1
        
         # normalize to [0, 1]
        # img = np.float32(img)
        # img = (img - self.mean) / self.std
        # img = (img - img.min()) / (img.max()-img.min()+0.0000000001)
    
        # mask = transform.resize(mask, 
        #                         (self.inp_size,self.inp_size), 
        #                         order=0,
        #                         preserve_range=True,
        #                         mode='constant',
        #                         anti_aliasing=False)
        mask = torch.from_numpy(np.uint8(mask)).unsqueeze(0) # tensor

        # img = transform.resize(img, 
        #                         (self.inp_size,self.inp_size), 
        #                         order=3,
        #                         preserve_range=True,
        #                         mode='constant',
        #                         anti_aliasing=False)
        img = torch.from_numpy(img) # to tensor不会改变维度方向[H, W, C]

        return {
            'image': img,
            'gt': mask,
            'original_size':self.inp_size,
        }


@register('test')
class TestDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset.test
        self.mode = dataset.mode


        self.inp_size = inp_size
        self.augment = augment


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, mask_path = self.dataset[idx]
        with open(img_path, 'rb') as f:
            img = pickle.load(f)
        with open(mask_path, 'rb') as f:
            mask = pickle.load(f)

        # get ndarray
        # if self.mode == 'npy':
        #     img =np.load(img_path) # array [x, y, c]
        #     mask = np.load(mask_path) # array [x, y]
        #     img = np.clip(img, self.H_min, self.H_max)
        #     img = (img - self.H_min) / (self.H_max - self.H_min) * 255.0 # ->[0, 255]
        # elif self.mode == 'rgb':
        #     img = Image.open(img_path)
        #     mask = Image.open(mask_path).convert('1')
        #     img = np.array(img)
        #     mask = np.array(mask)
        #     mask = np.uint8(mask)
        #     mask[mask==255]=1
        
         # normalize to [0, 1]
        # img = np.float32(img)
        # img = (img - self.mean) / self.std
        # img = (img - img.min()) / (img.max()-img.min()+0.0000000001)
    
        # mask = transform.resize(mask, 
        #                         (self.inp_size,self.inp_size), 
        #                         order=0,
        #                         preserve_range=True,
        #                         mode='constant',
        #                         anti_aliasing=False)
        mask = torch.from_numpy(np.uint8(mask)).unsqueeze(0) # tensor

        # img = transform.resize(img, 
        #                         (self.inp_size,self.inp_size), 
        #                         order=3,
        #                         preserve_range=True,
        #                         mode='constant',
        #                         anti_aliasing=False)
        img = torch.from_numpy(img) # to tensor不会改变维度方向[H, W, C]

        return {
            'image': img,
            'gt': mask,
            'original_size':self.inp_size,
        }
    

@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, inp_size=None,
                 augment=False, gt_resize=None):
        self.dataset = dataset.train
        self.mode = dataset.mode

        self.inp_size = inp_size
        self.augment = augment
        self.gt_resize = gt_resize
    

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, mask_path = self.dataset[idx]

        with open(img_path, 'rb') as f:
            img = pickle.load(f)
        with open(mask_path, 'rb') as f:
            mask = pickle.load(f)

        
        
         # normalize to [0, 1]
        # img = np.float32(img)
        # img = (img - self.mean) / self.std
        # img = (img - img.min()) / (img.max()-img.min()+0.0000000001)
    
        # mask = transform.resize(mask, 
        #                         (self.inp_size,self.inp_size), 
        #                         order=0,
        #                         preserve_range=True,
        #                         mode='constant',
        #                         anti_aliasing=False)
        mask = torch.from_numpy(np.uint8(mask)).unsqueeze(0) # tensor

        # img = transform.resize(img, 
        #                         (self.inp_size,self.inp_size), 
        #                         order=3,
        #                         preserve_range=True,
        #                         mode='constant',
        #                         anti_aliasing=False)
        img = torch.from_numpy(img) # to tensor不会改变维度方向[H, W, C]

        return {
            'image': img,
            'gt': mask,
            'original_size':self.inp_size,
        }
    