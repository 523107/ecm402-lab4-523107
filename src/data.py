import random
import numpy as np
import torch

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 523107
BATCH_SIZE = 128

DATA_DIR = "./data"

# Fashion-MNIST normalization values
MEAN = (0.2860,)
STD = (0.3530,)


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Device
# --------------------------------------------------

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Task 1(b): Transforms
# --------------------------------------------------

def get_transforms():

    # Training:
    # Augmentation + ToTensor + Normalization
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # Validation/Test:
    # NO augmentation
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    return train_transform, val_test_transform


# --------------------------------------------------
# Load Fashion-MNIST
# --------------------------------------------------

def load_fashion_mnist():

    train_transform, val_test_transform = get_transforms()

    # Dataset used to obtain the fixed train/validation indices
    base_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=None
    )

    # Official test set
    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=val_test_transform
    )

    return base_dataset, train_transform, val_test_transform, test_dataset


# --------------------------------------------------
# Create Train / Validation / Test DataLoaders
# --------------------------------------------------

def create_loaders():

    base_dataset, train_transform, val_test_transform, test_dataset = (
        load_fashion_mnist()
    )

    # 80/20 split
    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size

    generator = torch.Generator().manual_seed(SEED)

    train_subset, val_subset = random_split(
        base_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Create separate datasets so that augmentation
    # is applied ONLY to the training set.

    train_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=train_transform
    )

    val_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=val_test_transform
    )

    # Apply the same fixed indices obtained above
    train_dataset = torch.utils.data.Subset(
        train_dataset,
        train_subset.indices
    )

    val_dataset = torch.utils.data.Subset(
        val_dataset,
        val_subset.indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


# --------------------------------------------------
# Test the pipeline
# --------------------------------------------------

if __name__ == "__main__":

    set_seed()

    device = get_device()

    train_loader, val_loader, test_loader = create_loaders()

    print("Device:", device)
    print("Training samples:  ", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:      ", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)

    print("\nTask 1(b) transforms:")
    print("Training: RandomHorizontalFlip + RandomRotation + Normalize")
    print("Validation: No augmentation + Normalize")
    print("Test: No augmentation + Normalize")