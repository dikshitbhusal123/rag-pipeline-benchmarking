import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms
BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"

MODEL_PATH = BASE_DIR / "diagram_classifier.pth"

BATCH_SIZE = 32
Epochs = 30
learning_rate = 0.003
Image_Size = 224





print(f"Model saved to: {MODEL_PATH}")




device =torch.device("cpu")
print("training device:",device)


from torchvision import transforms

IMAGE_SIZE = 224

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

train_dataset = datasets.ImageFolder(
    root=BASE_DIR / "dataset" / "train",
    transform=train_transforms,
)

test_dataset = datasets.ImageFolder(
    root=BASE_DIR / "dataset" / "test",
    transform=test_transforms,
)
class_names = train_dataset.classes
print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))

assert train_dataset.classes == test_dataset.classes

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Detected Classes: {train_dataset.classes}")
print(f"Train images found: {len(train_dataset)} | Test images found: {len(test_dataset)}")



class MyNN(nn.Module):
    def __init__(self, input_features, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            # RGB images have 3 channels
            nn.Conv2d(input_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
           nn.Linear(64, num_classes)
        )
                

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
model = MyNN(input_features=3, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# training loop

for epoch in range(Epochs):

  total_epoch_loss = 0

  for batch_features, batch_labels in train_loader:

    # move data to gpu
    batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

    # forward pass
    outputs = model(batch_features)

    # calculate loss
    loss = criterion(outputs, batch_labels)

    # back pass
    optimizer.zero_grad()
    loss.backward()

    # update grads
    optimizer.step()

    total_epoch_loss = total_epoch_loss + loss.item()

  avg_loss = total_epoch_loss/len(train_loader)
  print(f"Epoch: {epoch + 1}/{Epochs}, Loss: {avg_loss:.4f}")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total

print(f"Final test accuracy: {test_accuracy:.2f}%")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "input_features": 3,
        "image_size": Image_Size,
    },
    MODEL_PATH,
)

print(f"Model saved to: {MODEL_PATH}")