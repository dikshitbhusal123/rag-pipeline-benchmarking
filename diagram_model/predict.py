
from pathlib import Path
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diagram_classifier.pth"
IMAGE_SIZE = 224

# Match the CNN architecture used in train.py
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

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.png")
        raise SystemExit(1)

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run python train.py first."
        )

    device = torch.device("cpu")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    class_names = checkpoint["class_names"]

    model = MyNN(
        input_features=checkpoint.get("input_features", 3),
        num_classes=len(class_names),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    with Image.open(image_path) as img:
        image = transform(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1)[0]

    confidence, predicted_index = probabilities.max(dim=0)

    print("Predicted category:", class_names[predicted_index.item()])
    print(f"Model confidence: {confidence.item() * 100:.2f}%")


if __name__ == "__main__":
    main()
