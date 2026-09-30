import matplotlib.pyplot as plt

from data import get_transforms
from torchvision import datasets


# --------------------------------------------------
# Load one original Fashion-MNIST image
# --------------------------------------------------

dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=None
)

image, label = dataset[0]

# Get training transform
train_transform, _ = get_transforms()


# --------------------------------------------------
# Create augmented versions
# --------------------------------------------------

augmented_1 = train_transform(image)
augmented_2 = train_transform(image)
augmented_3 = train_transform(image)


# --------------------------------------------------
# Display original + augmented images
# --------------------------------------------------

plt.figure(figsize=(10, 3))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(augmented_1.squeeze().numpy(), cmap="gray")
plt.title("Augmented 1")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(augmented_2.squeeze().numpy(), cmap="gray")
plt.title("Augmented 2")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(augmented_3.squeeze().numpy(), cmap="gray")
plt.title("Augmented 3")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "results/figures/task1b_augmentation.png",
    dpi=300
)

plt.show()