import torch
import torch.nn as nn
from torchvision import  transforms, models

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

# ----------- PARAMETRI GLOBALI ------------
IMG_SIZE = 224
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_SAVE_PATH = r'/ResNet/ResNet_18/resnet18_brain.pth'

# 🔹 Asigură-te că ai și funcția generate_gradcam(model, img_path) definită mai sus

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

def main():
    print("🔹 Încarc modelul ResNet-18 salvat...")
    # Creează arhitectura ResNet-18 fără weights predefinite (pentru a putea încărca ale tale)
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    # Încarcă parametrii antrenați
    state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    print("✅ Model ResNet-18 încărcat cu succes!")

    # Calea către imaginea de test
    img_path = r'/DataSet/archive/Testing/glioma/Te-gl_0036.jpg'
    print(f"🔹 Generez Grad-CAM pentru imaginea:\n{img_path}")

    # Apelează funcția Grad-CAM
    generate_gradcam(model, img_path)
    print("✅ Grad-CAM generat cu succes!")


if __name__ == "__main__":
    main()
