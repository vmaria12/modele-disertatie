import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from torchvision.models.vision_transformer import vit_l_16, ViT_L_16_Weights

# -------------------------
# PATHS
data_dir = r'C:\disi\DataSet\archive'
train_dir = os.path.join(data_dir, 'Training')
val_dir   = os.path.join(data_dir, 'Testing')

MODEL_SAVE_PATH = r'C:\disi\ViT_L_16\vit_l16_brain.pth'
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# -------------------------
# PARAMETERS
IMG_SIZE = 224
BATCH_SIZE = 4   # mic ca să nu dea OOM
EPOCHS = 3
LEARNING_RATE = 1e-4
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# -------------------------
# DATA TRANSFORMS
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# LOAD DATA
print("Loading datasets...")
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset   = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# -------------------------
# LOAD VIT-L/16 PRETRAINED
print("Loading ViT-L/16 model...")
weights = ViT_L_16_Weights.DEFAULT
model = vit_l_16(weights=weights)
num_ftrs = model.heads.head.in_features
model.heads.head = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(DEVICE)
print("Model loaded successfully.")

# -------------------------
# LOSS, OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------
# TRAIN LOOP
for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

    # -------------------------
    # VALIDATION
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    val_loss /= len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)
    print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")

# -------------------------
# SAVE MODEL
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved at: {MODEL_SAVE_PATH}")

# -------------------------
# ATTENTION MAP pentru ViT-L/16
def generate_vit_attention_map(model, img_path):
    model.eval()
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        class_idx = outputs.argmax(dim=1).item()

    # Atenție "fake" (placeholder)
    attn_map = np.random.rand(IMG_SIZE, IMG_SIZE)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)

    img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    plt.imshow(overlay)
    plt.title(f"Predicted: {CLASSES[class_idx]}")
    plt.axis('off')
    plt.show()

# -------------------------
# EXAMPLE USAGE:
# generate_vit_attention_map(model, r'C:\disi\DataSe_
