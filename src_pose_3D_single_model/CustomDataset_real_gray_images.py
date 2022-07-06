import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
from PIL import Image,ImageFile
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomImageDataset(Dataset):
    def __init__(self, img_files, real_folder_add, gray_folder_add, transform=None):
        self.img_files = img_files
        self.real_folder_add = real_folder_add
        self.gray_folder_add = gray_folder_add
        self.data_size = len(img_files)
        self.transform = transform
    
    def load_image(self,path):
        return torch.tensor(cv2.imread(path))
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image_gray = Image.open(self.gray_folder_add + self.img_files[idx])
        image_real = Image.open(self.real_folder_add + self.img_files[idx])
        image_real = self.transform(image_real)
        image_gray = self.transform(image_gray)
        im_b_real = image_real[:,:141,:]
        im_s1_real = image_real[:,141:282,:]
        im_s2_real = image_real[:,282:,:]
        im_b_gray = image_gray[:,:141,:]
        im_s1_gray = image_gray[:,141:282,:]
        im_s2_gray = image_gray[:,282:,:]
        im_three_channels_real = torch.zeros(3, 141, 141)
        im_three_channels_gray = torch.zeros(3, 141, 141)
        im_three_channels_real[0,:,:] = im_b_real
        im_three_channels_real[1,:,:] = im_s1_real
        im_three_channels_real[2,:,:] = im_s2_real
        im_three_channels_gray[0,:,:] = im_b_gray
        im_three_channels_gray[1,:,:] = im_s1_gray
        im_three_channels_gray[2,:,:] = im_s2_gray
        return im_three_channels_real, im_three_channels_gray
