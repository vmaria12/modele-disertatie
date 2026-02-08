import os
import glob
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.optim as optim

try:
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
except ImportError:
    print("Eroare: Biblioteca segment-anything nu este instalata)
    exit(1)

# Configurări
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = r"D:\Disertatie\BE-disertatie\djangoProject1\Models\SAM\sam_b.pt"
DATASET_DIR = r"D:\Disertatie\DataSet\demo" 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

class SAMDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # Căutăm recursiv toate imaginile .jpg
        self.image_paths = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)
        self.transform = transform
        
        print(f"Au fost găsite {len(self.image_paths)} imagini în {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # SAM așteaptă RGB

        # Masca
        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (int(w/2), int(h/2))
        radius = int(min(h, w) / 4)
        cv2.circle(mask, center, radius, 1, -1) 

        # Box
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            box = np.array([x_min, y_min, x_max, y_max])
        else:
            box = np.array([0, 0, w, h]) 

        # Pregătire date pentru SAM
        if self.transform:
            # Resize imagine și creare embedding
            image = self.transform.apply_image(image)
            # Resize box
            box = self.transform.apply_boxes(box[None, :], (h, w)).squeeze(0)
            
            # Convert to torch tensor
            image_torch = torch.as_tensor(image).permute(2, 0, 1).float() 
            mask_torch = torch.as_tensor(mask).float().unsqueeze(0) # [1, H, W]
            box_torch = torch.as_tensor(box).float()    
            return image_torch, mask_torch, box_torch, (h, w)

        return image, mask, box

import torch.nn as nn

# --- NOU: CLASA ADAPTER ---
class Adapter(nn.Module):
    def __init__(self, model_dim, bottleneck_dim, activation=nn.GELU()):
        super().__init__()
        self.down_project = nn.Linear(model_dim, bottleneck_dim)
        self.activation = activation
        self.up_project = nn.Linear(bottleneck_dim, model_dim)
        # Initialize Up projection weights to zero or near-zero
        # helps stabilize training early on
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        # x is the input hidden state (e.g., output of MHA or FFN)
        down = self.down_project(x)
        activated = self.activation(down)
        up = self.up_project(activated)
        # Add residual connection
        output = x + up
        return output

def apply_adapters(sam_model, bottleneck_dim=64):
    """
    Injectează module Adapter în blocurile ViT (Attention și MLP) ale encoder-ului SAM.
    Acest lucru transformă encoder-ul într-un "Adapter ViT" complet.
    """
    print(f"Injectare Adaptere (Attn + MLP) cu bottleneck_dim={bottleneck_dim}...")
    
    #  (ViT blocks)
    for i, block in enumerate(sam_model.image_encoder.blocks):
        # --- 1. Adapter pentru MLP ---
        model_dim_mlp = block.mlp.lin1.in_features
        adapter_mlp = Adapter(model_dim_mlp, bottleneck_dim)
        
        #  MLP-ul: MLP(x) -> Adapter(MLP(x))
        block.mlp = nn.Sequential(
            block.mlp,
            adapter_mlp
        )

        # --- 2. Adapter pentru Attention ---
        model_dim_attn = block.mlp.lin1.in_features 
        adapter_attn = Adapter(model_dim_attn, bottleneck_dim)
        
        # Attn(x) -> Adapter(Attn(x))
        block.attn = nn.Sequential(
            block.attn,
            adapter_attn
        )
    
    print(f"Adaptere adăugate cu succes (Attn & MLP) în {len(sam_model.image_encoder.blocks)} blocuri.")

def main():
    print(f"Rulează pe dispozitiv: {DEVICE}")

    # 1. Inițializare Model SAM
    print("Se încarcă modelul SAM...")
    sam_model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam_model.to(DEVICE)

    # 2. Aplicare Adaptere
    apply_adapters(sam_model, bottleneck_dim=64)
    
    # 3. Configurare gradienți (Freezing Strategy)
    for param in sam_model.image_encoder.parameters():
        param.requires_grad = False
        
    # Activăm gradienții DOAR pentru Adaptere
    for block in sam_model.image_encoder.blocks:
        for param in block.mlp[1].parameters():
            param.requires_grad = True
        
        for param in block.attn[1].parameters():
            param.requires_grad = True

    # Prompt Encoder rămâne înghețat
    for param in sam_model.prompt_encoder.parameters():
        param.requires_grad = False

    # Mask Decoder antrenabil (opțional, dar recomandat pentru fine-tuning puternic)
    for param in sam_model.mask_decoder.parameters():
        param.requires_grad = True

    # Verificare
    trainable_params = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in sam_model.parameters())
    print(f"Parametri antrenabili: {trainable_params} / {total_params} ({trainable_params/total_params:.2%})")

    # 4. Dataset și DataLoader
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    dataset = SAMDataset(root_dir=DATASET_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # 5. Optimizator și Loss
    # Adunăm toți parametrii care necesită gradienți (Adaptere + Mask Decoder)
    trainable_parameters = [p for p in sam_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_parameters, lr=LEARNING_RATE, weight_decay=1e-4) # Loss functions și bucla de antrenare urmează...
    
    loss_fn = torch.nn.MSELoss() 

    # 6. Bucla de Antrenare
    print("Începe antrenarea (Fine-tuning cu Adaptere)...")
    sam_model.train()
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for i, (images, gt_masks, boxes, original_sizes) in enumerate(dataloader):
            images = images.to(DEVICE)     
            boxes = boxes.to(DEVICE)       
            gt_masks = gt_masks.to(DEVICE) 

            # SAM input structure
            input_dict_list = []
            current_batch_size = images.shape[0]
            
            for k in range(current_batch_size):
                 original_h = original_sizes[0][k].item()
                 original_w = original_sizes[1][k].item()
                 
                 input_dict_list.append({
                     'image': images[k], 
                     'boxes': boxes[k].unsqueeze(0), 
                     'original_size': (original_h, original_w)
                 })

            outputs = sam_model(input_dict_list, multimask_output=False)
            
            total_loss = 0
            for k, output in enumerate(outputs):
                pred_mask = output['masks']
                gt_mask = gt_masks[k].unsqueeze(0) 
                
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = F.interpolate(pred_mask, size=gt_mask.shape[-2:], mode='bilinear', align_corners=False)
                
                pred_probs = torch.sigmoid(pred_mask)
                loss = loss_fn(pred_probs, gt_mask)
                total_loss += loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {total_loss.item()/current_batch_size:.4f}")

        print(f"--- Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader):.4f} ---")
        
        if (epoch + 1) % 5 == 0:
            save_path = f"sam_adapter_finetuned_epoch_{epoch+1}.pth"
            torch.save({
                'model_state_dict': sam_model.state_dict(),
                'adapter_config': {'bottleneck_dim': 64}
            }, save_path)
            print(f"Model (checkpoint) salvat la: {save_path}")

    print("Antrenare completă!")

if __name__ == '__main__':
    main()
