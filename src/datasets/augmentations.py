"""
===========================================================
PapsAI XNet Image Augmentations
===========================================================

Image augmentation pipeline used during CNN training.

The implementation follows the methodology described in the
Nature Scientific Reports manuscript.

Augmentations

• Random Rotation (±20°)
• Horizontal Flip
• Vertical Flip
• Brightness Adjustment
• Contrast Adjustment
• Gaussian Noise

Augmentations are applied ONLY to the training dataset.
===========================================================
"""
import random

import numpy as np

import torch

from PIL import Image

from torchvision import transforms
import random

import numpy as np

from PIL import Image

from torchvision import transforms


############################################################

class AddGaussianNoise(object):
    """
    Add Gaussian noise to an image.

    Parameters
    ----------
    mean : float

    std : float

    probability : float
    """

    def __init__(
            self,
            mean=0.0,
            std=0.02,
            probability=0.5):

        self.mean = mean
        self.std = std
        self.probability = probability

    ########################################################

    def __call__(self, tensor):

        if random.random() > self.probability:
            return tensor

        noise = torch.randn_like(tensor)

        noise = noise * self.std + self.mean

        tensor = tensor + noise

        tensor = torch.clamp(
            tensor,
            0.0,
            1.0
        )

        return tensor


############################################################

def training_transforms():

    """
    Data augmentation used during CNN training.
    """

    return transforms.Compose([

        transforms.Resize(
            (224,224)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomVerticalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=20
        ),

        transforms.ColorJitter(

            brightness=0.20,

            contrast=0.20

        ),

        transforms.ToTensor(),

        AddGaussianNoise(

            mean=0.0,

            std=0.02,

            probability=0.50

        ),

        transforms.Normalize(

            mean=[0.485,0.456,0.406],

            std=[0.229,0.224,0.225]

        )

    ])


############################################################

def validation_transforms():

    """
    Validation preprocessing.

    No augmentation.
    """

    return transforms.Compose([

        transforms.Resize(

            (224,224)

        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.485,0.456,0.406],

            std=[0.229,0.224,0.225]

        )

    ])


############################################################

def testing_transforms():

    """
    Testing preprocessing.

    No augmentation.
    """

    return validation_transforms()


############################################################

def inference_transforms():

    """
    Inference preprocessing.

    Used by inference.py.
    """

    return validation_transforms()
