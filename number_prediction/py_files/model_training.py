# Import PyTorch main library
import os
from pyexpat import model
import torch
# Import neural network module (layers, activations, losses, etc.)
import torch.nn as nn
# Import optimizer module (Adam, SGD, etc.)
import torch.optim as optim
# Import DataLoader for batching, shuffling, and loading datasets
from torch.utils.data import DataLoader
# Import torchvision datasets (MNIST) and transform utilities
from torchvision import datasets, transforms


# ----------------------------
# 1. Dataset & Dataloader
# ----------------------------
print("Setting up dataset transforms...")

# Define image transformations:
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Loading MNIST training dataset...")
train_loader = DataLoader(
    datasets.MNIST(root="./number_prediction", train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

print("Loading MNIST test dataset...")
test_loader = DataLoader(
    datasets.MNIST(root="./number_prediction", train=False, download=True, transform=transform),
    batch_size=64
)

print("Datasets and dataloaders ready.\n")


# ----------------------------
# 2. Improved CNN
# ----------------------------
print("Building CNN model...")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

print("Model created.\n")


# ----------------------------
# 3. Training Function
# ----------------------------
def train_model(epochs=10):

    print("Checking for GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("Initializing model...")
    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Loss function and optimizer ready.\n")

    print("Starting training...\n")

    # Loop through the number of training epochs.
    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        model.train()
        total_loss = 0

        batch_index = 0

        for images, labels in train_loader:
            batch_index += 1

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("\nTraining step completed. Running evaluation...")

        # Evaluation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        print(f"Epoch {epoch+1} finished.")
        print(f"Total Loss: {total_loss:.4f}")
        print(f"Test Accuracy: {correct/total*100:.2f}%\n")

    print("Training completed. Saving model...")

    current_folder = os.path.dirname(os.path.abspath(__file__))
    parent_folder = os.path.dirname(current_folder)

    os.makedirs(parent_folder, exist_ok=True)

    save_path = os.path.join(parent_folder, "model.pth")

    # Save the model
    torch.save(model.state_dict(), save_path)

    print(f"Model saved to {save_path}")
    print("Model saved as model.pth\n")


# ----------------------------
# 4. Main
# ----------------------------
if __name__ == "__main__":
    print("Program started.\n")
    train_model()
    print("Program finished.")
