import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from glob import glob
import ntpath

class MRIDataset(Dataset):
    def __init__(self, args, data_path , transform = None, mode = 'Training', plane = False):

        dirs = glob(os.path.join(data_path, '*'))
        images_path = [glob(os.path.join(x, '*.tif')) for x in dirs]
        images_path = [item for sublist in images_path for item in sublist]
        original_images_path = []

        for item in images_path:
            if '_mask' not in item:
                original_images_path.append(item)

        original_masks_path = []
        for item in original_images_path:
            original_masks_path.append(item[:-4] + '_mask.tif')

        self.name_list = original_images_path
        self.label_list = original_masks_path
        self.data_path = data_path
        self.mode = mode

        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def path_leaf(path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = name
        
        msk_path = self.label_list[index]

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        return (img, mask, name)