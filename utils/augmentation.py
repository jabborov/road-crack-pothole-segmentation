import numpy as np
import random

import torch
import torchvision.transforms.functional as TF

class Augmentation:
    """Standard Augmentation"""   

    def __init__(self, flip_prop: float = 0.5) -> None:
        transforms = []
        if flip_prop > 0:
            transforms.append(ImgMaskHorizontalFlip(flip_prop))
            transforms.append(ImagMaskVerticalFlip(flip_prop))           
        transforms.extend([ImgMaskToTensor(), ConvertImageDtype(torch.float)])
        self.transforms = Compose(transforms)

    def __call__(self, img, mask):
        return self.transforms(img, mask)
    
class ImgMaskToTensor:
    """Convert PIL image to torch tensor"""    

    def __call__(self, image, mask):        
        image = TF.pil_to_tensor(image)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ImagMaskVerticalFlip:
    """Apply vertical flips to image and mask."""

    def __init__(self, p: float=0.5):
        self.p = p

    def __call__(self, image, mask):
        p = random.random()
        if p < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)        
        return image, mask 

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'

class ImgMaskHorizontalFlip:
    """Apply horizontal flips to image and mask."""

    def __init__(self, p: float=0.5):
        self.p = p

    def __call__(self, image, mask):
        p = random.random()
        if p < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)        
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'

class Compose:
    """Compose all transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):              
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
        

class ConvertImageDtype:
    """Convert Image dtype"""

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, mask):
        image = TF.convert_image_dtype(image, self.dtype)
        return image, mask



