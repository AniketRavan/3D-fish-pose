import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
from PIL import Image
import cv2

class CustomImageDataset(Dataset):
    def __init__(self, img_files_address, pose_files_address, eye_coor_files_add, crop_coor_files_add, transform=None):
        self.img_files_address = img_files_address
        self.pose_files_address = pose_files_address
        self.eye_coor_files_add = eye_coor_files_add
        self.crop_coor_files_add = crop_coor_files_add
        self.data_size = len(img_files_address)
        self.transform = transform
    
    def load_image(self,path):
        return torch.tensor(cv2.imread(path))
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image = Image.open(self.img_files_address[idx])
        image = self.transform(image)
        im_b = image[:,:141,:]
        im_s1 = image[:,141:282,:]
        im_s2 = image[:,282:,:]
        im_three_channels = torch.zeros(3, 141, 141)
        im_three_channels[0,:,:] = im_b
        im_three_channels[1,:,:] = im_s1
        im_three_channels[2,:,:] = im_s2
        pose = torch.load(self.pose_files_address[idx])
        eye_coor = torch.load(self.eye_coor_files_add[idx])
        crop_coor = torch.load(self.crop_coor_files_add[idx])
        return im_three_channels,pose,eye_coor,crop_coor
