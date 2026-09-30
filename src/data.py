import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


SEED = 523107
BATCH_SIZE = 128


def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Use GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    """Create training and validation transforms."""

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    return train_transform, val_transform


def load_fashion_mnist():
    """
    Download Fashion-MNIST and create
    80% training / 20% validation splits.
    """

    train_transform, val_transform = get_transforms()

    # Download the original training dataset.
    full_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=None
    )

    # Fixed split for reproducibility.
    generator = torch.Generator().manual_seed(SEED)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_indices, val_indices = random_split(
        range(len(full_dataset)),
        [train_size, val_size],
        generator=generator
    )

    # Separate datasets so augmentation is applied only to training.
    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=False,
        transform=train_transform
    )

    val_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=False,
        transform=val_transform
    )

    train_dataset = torch.utils.data.Subset(
        train_dataset,
        train_indices.indices
    )

    val_dataset = torch.utils.data.Subset(
        val_dataset,
        val_indices.indices
    )

    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=val_transform
    )

    return train_dataset, val_dataset, test_dataset


def create_loaders():
    """Create DataLoaders for train, validation and test sets."""

    train_dataset, val_dataset, test_dataset = load_fashion_mnist()

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


if __name__ == "__main__":
    set_seed()

    device = get_device()

    train_loader, val_loader, test_loader = create_loaders()

    print("=" * 60)
    print("LAB 4 DATA PIPELINE")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Training samples:   {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples:       {len(test_loader.dataset)}")

    images, labels = next(iter(train_loader))

    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")