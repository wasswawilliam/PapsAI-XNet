"""
===========================================================
Herlev Dataset Loader
===========================================================

Dataset loader for PapsAI XNet.

This module implements the data pipeline described in the
manuscript:

- Herlev Pap-smear dataset
- Seven cervical cytology classes
- Stratified 70/10/20 train-validation-test split
- Training-only augmentation
- Validation and test preprocessing without augmentation
- RGB conversion
- 224 x 224 image resizing
- ImageNet normalization
- Reproducible random seed
===========================================================
"""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from .augmentations import (
    training_transforms,
    validation_transforms,
    testing_transforms,
)


CLASS_NAMES = [
    "Superficial Squamous Epithelial",
    "Intermediate Squamous Epithelial",
    "Columnar Epithelial",
    "Mild Dysplasia",
    "Moderate Dysplasia",
    "Severe Dysplasia",
    "Carcinoma in Situ",
]


CLASS_MAPPING = {
    "superficial_squamous_epithelial": 0,
    "intermediate_squamous_epithelial": 1,
    "columnar_epithelial": 2,
    "mild_dysplasia": 3,
    "moderate_dysplasia": 4,
    "severe_dysplasia": 5,
    "carcinoma_in_situ": 6,
}


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HerlevDataset(Dataset):
    """
    PyTorch Dataset for the Herlev Pap-smear dataset.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        split: str = "train",
    ):
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length.")

        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be one of: train, validation, test.")

        self.image_paths = image_paths
        self.labels = labels
        self.split = split

        if split == "train":
            self.transforms = training_transforms()
        elif split == "validation":
            self.transforms = validation_transforms()
        else:
            self.transforms = testing_transforms()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)

        label = int(self.labels[index])

        return image, label


def discover_images(dataset_root: str) -> Tuple[List[str], List[int]]:
    """
    Discover image paths and labels from a class-folder dataset structure.

    Expected structure:

    datasets/Herlev/
        superficial_squamous_epithelial/
        intermediate_squamous_epithelial/
        columnar_epithelial/
        mild_dysplasia/
        moderate_dysplasia/
        severe_dysplasia/
        carcinoma_in_situ/
    """

    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    image_paths: List[str] = []
    labels: List[int] = []

    for class_folder, class_index in CLASS_MAPPING.items():
        folder = dataset_root / class_folder

        if not folder.exists():
            raise FileNotFoundError(
                f"Expected class folder not found: {folder}"
            )

        for image_file in sorted(folder.iterdir()):
            if image_file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                image_paths.append(str(image_file))
                labels.append(class_index)

    if len(image_paths) == 0:
        raise RuntimeError(
            f"No image files found in dataset root: {dataset_root}"
        )

    return image_paths, labels


def create_splits(
    dataset_root: str,
    random_seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.20,
):
    """
    Create stratified train, validation, and test datasets.

    Default split:
        Train: 70%
        Validation: 10%
        Test: 20%
    """

    total = train_ratio + validation_ratio + test_ratio

    if not np.isclose(total, 1.0):
        raise ValueError(
            "train_ratio + validation_ratio + test_ratio must equal 1.0"
        )

    set_seed(random_seed)

    image_paths, labels = discover_images(dataset_root)

    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=labels,
    )

    validation_fraction_of_train_val = validation_ratio / (
        train_ratio + validation_ratio
    )

    train_images, validation_images, train_labels, validation_labels = train_test_split(
        train_val_images,
        train_val_labels,
        test_size=validation_fraction_of_train_val,
        random_state=random_seed,
        stratify=train_val_labels,
    )

    train_dataset = HerlevDataset(
        train_images,
        train_labels,
        split="train",
    )

    validation_dataset = HerlevDataset(
        validation_images,
        validation_labels,
        split="validation",
    )

    test_dataset = HerlevDataset(
        test_images,
        test_labels,
        split="test",
    )

    return train_dataset, validation_dataset, test_dataset


def create_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    random_seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    """

    train_dataset, validation_dataset, test_dataset = create_splits(
        dataset_root=dataset_root,
        random_seed=random_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, validation_loader, test_loader


def get_class_distribution(labels: List[int]) -> dict:
    """
    Return class distribution for a list of labels.
    """

    distribution = {}

    for class_index, class_name in enumerate(CLASS_NAMES):
        distribution[class_name] = int(np.sum(np.array(labels) == class_index))

    return distribution


def summarize_dataset(dataset: HerlevDataset) -> dict:
    """
    Summarize a HerlevDataset object.
    """

    return {
        "split": dataset.split,
        "num_samples": len(dataset),
        "class_distribution": get_class_distribution(dataset.labels),
    }
