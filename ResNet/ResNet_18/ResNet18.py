import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


# -------------------------
# PATHS
data_dir = r'/DataSet/archive'
train_dir = os.path.join(data_dir, 'Training')
val_dir   = os.path.join(data_dir, 'Testing')


MODEL_SAVE_PATH = r'/ResNet/ResNet_18/resnet18_brain.pth'
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


# -------------------------
# PARAMETERS
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-4
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------------------------
# DATA TRANSFORMS
train_transforms = transforms.Compose([
   transforms.Resize((IMG_SIZE, IMG_SIZE)),
   transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406],  # mean std pentru ImageNet
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
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset   = datasets.ImageFolder(val_dir, transform=val_transforms)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -------------------------
# LOAD RESNET18 PRETRAINED
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)  # adaptăm pentru 4 clase
model = model.to(DEVICE)


# -------------------------
# LOSS, OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# -------------------------
# TRAIN LOOP
for epoch in range(EPOCHS):
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
   print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")


   # -------------------------
   # VALIDATION
   model.eval()
   val_corrects = 0
   val_loss = 0.0
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
   print(f"Validation - Loss: {val_loss:.4f} - Acc: {val_acc:.4f}")


# -------------------------
# SAVE MODEL
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model salvat în: {MODEL_SAVE_PATH}")


# -------------------------
# GRAD-CAM IMPLEMENTATION00
def generate_gradcam(model, img_path, target_layer_name='layer4'):
   model.eval()
   img = Image.open(img_path).convert('RGB')
   transform = transforms.Compose([
       transforms.Resize((IMG_SIZE, IMG_SIZE)),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
   ])
   img_tensor = transform(img).unsqueeze(0).to(DEVICE)


   # Hook pentru activări și gradient
   activations = []
   gradients = []


   def forward_hook(module, input, output):
       activations.append(output)


   def backward_hook(module, grad_in, grad_out):
       gradients.append(grad_out[0])


   # Atașează hook pe ultimul strat de convoluție
   for name, module in model.named_modules():
       if name == target_layer_name:
           module.register_forward_hook(forward_hook)
           module.register_backward_hook(backward_hook)


   # Forward + backward
   outputs = model(img_tensor)
   class_idx = outputs.argmax(dim=1).item()
   score = outputs[0, class_idx]
   model.zero_grad()
   score.backward()


   # Calcul Grad-CAM
   grads = gradients[0].mean(dim=[2, 3], keepdim=True)
   activation = activations[0]
   cam = (activation * grads).sum(dim=1, keepdim=True)
   cam = torch.relu(cam)
   cam = cam.squeeze().cpu().detach().numpy()
   cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
   cam = (cam - cam.min()) / (cam.max() - cam.min())


   # Vizualizare
   img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
   heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
   overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
   plt.imshow(overlay)
   plt.title(f"Predicted: {CLASSES[class_idx]}")
   plt.axis('off')
   plt.show()


# -------------------------
# EXAMPLE USAGE:

def main():
    print("🔹 Încarc modelul ResNet-34 salvat...")
    model = models.resnet34(weights=None)  # inițial fără weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    print("✅ Model încărcat cu succes!")

    # imagine pentru Grad-CAM
    img_path = r'/DataSet/archive/Training/glioma/Tr-gl_0010.jpg'
    print(f"🔹 Generez Grad-CAM pentru imaginea:\n{img_path}")

    generate_gradcam(model, img_path)
    print("✅ Grad-CAM generat cu succes!")
if __name__ == "__main__":
    main()
