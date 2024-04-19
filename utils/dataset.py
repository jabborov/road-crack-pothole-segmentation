import numpy as np
from PIL import Image
import os
import random

import torch
from torch.utils.data import Dataset

from utils.augmentation import Augmentation

def random_seed(seed=42):
    """Random Number Generators"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class Crack(Dataset):
    def __init__(
        self, root: str, image_size: int = 448, transforms: Augmentation = Augmentation(), mask_suffix: str = "_mask")-> None:
        self.root = root
        self.image_size = image_size
        self.mask_suffix = mask_suffix
        self.filenames = [os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(self.root, "images"))]
        if not self.filenames:
            raise FileNotFoundError(f"Files not found in {root}")
        self.transforms = transforms
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        
        image_path = os.path.join(self.root, f"images{os.sep}{filename}.jpg")
        mask_path = os.path.join(self.root, f"masks{os.sep}{filename + self.mask_suffix}.jpg")

        # image and mask loading
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # resizing
        image = image.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.BICUBIC)        
        
        assert image.size == mask.size, f"`image`: {image.size} and `mask`: {mask.size} should be the same size, but are {image.size} and  {mask.size}"  

        """TODO: The mask image color range should be [0, 1]. In Pothole and Road Crack datasets the mask image includes values between 0 and 255, however
        it was supposed to be 0 and 1. So mask image divided by 255 to make it between 0 and 1."""
        if (np.asarray(mask) > 1).any():
            mask = np.asarray(np.asarray(mask) / 255.0, dtype=np.byte)
            mask = Image.fromarray(mask)
        
        # transform
        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask
