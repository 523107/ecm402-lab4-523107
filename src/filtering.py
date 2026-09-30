import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data import load_fashion_mnist


# --------------------------------------------------
# Fixed 3x3 filters
# --------------------------------------------------

sobel_x = torch.tensor(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]],
    dtype=torch.float32
).view(1, 1, 3, 3)

sobel_y = torch.tensor(
    [[-1, -2, -1],
     [ 0,  0,  0],
     [ 1,  2,  1]],
    dtype=torch.float32
).view(1, 1, 3, 3)

gaussian = torch.tensor(
    [[1, 2, 1],
     [2, 4, 2],
     [1, 2, 1]],
    dtype=torch.float32
).view(1, 1, 3, 3) / 16.0


# --------------------------------------------------
# Apply filters using conv2d
# --------------------------------------------------

def apply_filters(image):
    # image shape: [1, 1, 28, 28]

    sobel_x_result = F.conv2d(
        image,
        sobel_x,
        padding=1
    )

    sobel_y_result = F.conv2d(
        image,
        sobel_y,
        padding=1
    )

    gaussian_result = F.conv2d(
        image,
        gaussian,
        padding=1
    )

    return sobel_x_result, sobel_y_result, gaussian_result


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    # Load Fashion-MNIST
    _, _, test_dataset = load_fashion_mnist()

    # Take one image
    image, label = test_dataset[0]

    # Add batch dimension
    image = image.unsqueeze(0)

    # Apply filters
    sobel_x_result, sobel_y_result, gaussian_result = apply_filters(image)

    # Print information
    print("Original image shape:", image.shape)
    print("Sobel-X shape:", sobel_x_result.shape)
    print("Sobel-Y shape:", sobel_y_result.shape)
    print("Gaussian shape:", gaussian_result.shape)
    print("Class label:", label)

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    plt.figure(figsize=(12, 3))

    plt.subplot(1, 4, 1)
    plt.imshow(image[0, 0].numpy(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(sobel_x_result[0, 0].detach().numpy(), cmap="gray")
    plt.title("Sobel-X")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(sobel_y_result[0, 0].detach().numpy(), cmap="gray")
    plt.title("Sobel-Y")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(gaussian_result[0, 0].detach().numpy(), cmap="gray")
    plt.title("Gaussian Blur")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("results/figures/task1a_filters.png", dpi=300)
    plt.show()