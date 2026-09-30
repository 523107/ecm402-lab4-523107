import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data import create_loaders, get_device, set_seed
from models import CustomCNN


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 523107
EPOCHS = 15
LEARNING_RATE = 0.001


# --------------------------------------------------
# Training function
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# Validation function
# --------------------------------------------------

def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # No gradients needed during validation
    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# Plot training curves
# --------------------------------------------------

def plot_training_curves(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies
):

    epochs = range(1, len(train_losses) + 1)

    # Loss curve
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        "results/figures/task2_loss_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # Accuracy curve
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accuracies,
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracies,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        "results/figures/task2_accuracy_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    # Reproducibility
    set_seed(SEED)

    # Device
    device = get_device()

    print("Device:", device)

    # Data
    train_loader, val_loader, test_loader = create_loaders()

    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))

    # Model
    model = CustomCNN().to(device)

    print("\nModel:")
    print(model)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Training history
    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    best_val_accuracy = 0.0
    best_epoch = 0

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    print("\nStarting training...\n")

    total_training_start = time.time()

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )

        epoch_time = time.time() - epoch_start

        # Store results
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Track best validation model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                "results/best_custom_cnn.pth"
            )

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = time.time() - total_training_start

    # --------------------------------------------------
    # Final training summary
    # --------------------------------------------------

    print("\nTraining completed.")

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.2f}%"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Average time per epoch: "
        f"{total_training_time / EPOCHS:.2f}s"
    )

    print(
        f"Total training time: "
        f"{total_training_time:.2f}s"
    )

    # --------------------------------------------------
    # Plot curves
    # --------------------------------------------------

    plot_training_curves(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies
    )