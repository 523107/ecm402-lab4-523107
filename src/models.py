import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Input: 1 x 28 x 28
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            # 32 x 28 x 28 -> 32 x 14 x 14
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            # 64 x 14 x 14 -> 64 x 7 x 7
            nn.MaxPool2d(kernel_size=2),

            nn.Dropout(p=0.25)
        )

        # Classification
        self.classifier = nn.Sequential(
            nn.Flatten(),

            # 64 x 7 x 7 = 3136
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),

            nn.Dropout(p=0.5),

            # 10 Fashion-MNIST classes
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# --------------------------------------------------
# Test the model
# --------------------------------------------------

if __name__ == "__main__":

    model = CustomCNN()

    print(model)

    # Count trainable parameters
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTrainable parameters:", trainable_params)

    # Test with one batch
    x = torch.randn(128, 1, 28, 28)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)