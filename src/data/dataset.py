import os
import cv2
from PIL import Image
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.data.preprocess import base_transform

PATH = '../data/raw'
class SourceDataset(Dataset):
    """
    Torch data set object for the source domain data
    """
    def __init__(self, csv_file, transform=base_transform, is_training=True):
        """
        :param csv_file: path for meta data 
        :param node_cfg_dataset: the DATASET node of the config
        :param is_training: choose 
        """
        self.csv_file = os.path.join(PATH, csv_file)
        self.data = pd.read_csv(self.csv_file)   # meta data
        self.transform = transform          # preprocessing
        self.is_training = is_training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(PATH,self.data.iloc[idx, 1])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Inference mode에서는 transform된 이미지를 출력
        if not self.is_training:
            if self.transform:
                image = self.transform(image=image)['image']
            return image
        

        mask_path = os.path.join(PATH, self.data.iloc[idx, 2])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12 #배경을 픽셀값 12로 간주

        #
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

            return image, mask
    

class TargetDataset(Dataset):
    """
    Torch data set object for the source domain data
    """
    def __init__(self, csv_file, transform=base_transform, is_training=True):
        """
        :param csv_file: path for meta data 
        :param node_cfg_dataset: the DATASET node of the config
        :param is_training: choose 
        """
        self.csv_file = os.path.join(PATH, csv_file)
        self.data = pd.read_csv(self.csv_file)   # meta data
        self.transform = transform          # preprocessing
        self.infer = not is_training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(PATH,self.data.iloc[idx, 1])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Inference mode에서는 transform된 이미지를 출력
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image
        

        mask_path = self.data.iloc[idx, 2]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12 #배경을 픽셀값 12로 간주

        #
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

            return image, mask
    
    