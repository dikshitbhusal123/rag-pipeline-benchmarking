
from pathlib import Path

import fitz
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diagram_model" / "diagram_classifier.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cpu")



class MyNN(nn.Module):
    def __init__(self, input_features=3, num_classes=3):
        super().__init__()


        self.features = nn.Sequential(
            nn.Conv2d(input_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )


        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


    

            
     


if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

class_names = checkpoint["class_names"]

model = MyNN(
    input_features=checkpoint["input_features"],
    num_classes=len(class_names)
)

model.load_state_dict(checkpoint["model_state_dict"])
print("Loading CNN checkpoint from:", MODEL_PATH)
print("Checkpoint classifier shape:", checkpoint["model_state_dict"]["classifier.2.weight"].shape)
model.eval()

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def classify_page(page):
    """Render one PDF page and classify the entire page image."""
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    index = probabilities.argmax().item()

    return {
        "category": class_names[index],
        "confidence": float(probabilities[index].item()),
    }


def classify_pdf(pdf_path):
    """Return one CNN prediction for each page in a PDF."""
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    results = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            prediction = classify_page(page)
            results.append({
                "page": page_number,
                **prediction,
            })

    return results
