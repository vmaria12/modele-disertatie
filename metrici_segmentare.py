import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO, SAM
from tqdm import tqdm
from scipy.spatial.distance import cdist

# --- Configurare Plotting ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def select_directory(title="Selectați directorul"):
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title=title)
    return folder_selected

def select_file(title="Selectați fișierul", filetypes=[("All Files", "*.*")]):
    root = tk.Tk()
    root.withdraw()
    file_selected = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_selected

def compute_dice_iou(mask_true, mask_pred):
    """Calculează Dice Coefficient și IoU."""
    # Asigurare că sunt binare (0 și 1)
    mask_true = (mask_true > 0).astype(np.uint8)
    mask_pred = (mask_pred > 0).astype(np.uint8)

    intersection = np.sum(mask_true * mask_pred)
    sum_true = np.sum(mask_true)
    sum_pred = np.sum(mask_pred)

    dice = (2. * intersection) / (sum_true + sum_pred + 1e-6)
    iou = intersection / (sum_true + sum_pred - intersection + 1e-6)

    return dice, iou

def compute_tumor_size(mask):
    """Calculează dimensiunea tumorii (număr pixeli)."""
    return np.sum(mask > 0)

def compute_curvature(mask):
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0

    # Luăm cel mai mare contur (presupunând că e tumora principală)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if area == 0:
        return 0
    
    complexity = (perimeter ** 2) / (4 * np.pi * area)
    
    return complexity

def run_evaluation():
    print("=== Script Evaluare Segmentare SAM pe TCGA Dataset ===")

    # 1. Selectare Resurse
    print("\n[Pasul 1] Selectați modelul YOLO pentru detecție (Box Prompt)...")
    yolo_path = select_file("Selectați modelul YOLO (.pt)", [("YOLO Model", "*.pt")])
    if not yolo_path: return

    print("\n[Pasul 2] Selectați modelul SAM pentru segmentare...")
    sam_path = select_file("Selectați modelul SAM (.pt)", [("SAM Model", "*.pt")])
    if not sam_path: return

    print("\n[Pasul 3] Selectați directorul cu IMAGINI de test (TCGA)...")
    images_dir = select_directory("Director Imagini Test")
    if not images_dir: return

    print("\n[Pasul 4] Selectați directorul cu MĂȘTI Ground Truth...")
    masks_dir = select_directory("Director Măști Ground Truth")
    if not masks_dir: return

    # Inițializare Modele
    print(f"\nÎncărcare modele...\n YOLO: {os.path.basename(yolo_path)}\n SAM: {os.path.basename(sam_path)}")
    try:
        yolo_model = YOLO(yolo_path)
        sam_model = SAM(sam_path)
    except Exception as e:
        print(f"Eroare la încărcarea modelelor: {e}")
        return

    results = []
    
    # Listare imagini
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    
    print(f"\nÎncepere evaluare pe {len(image_files)} imagini...")

    for img_name in tqdm(image_files):
        img_path = os.path.join(images_dir, img_name)
        
        mask_name = img_name
        mask_path = os.path.join(masks_dir, mask_name)
        
        if not os.path.exists(mask_path):
            base_name = os.path.splitext(img_name)[0]
            potential_masks = [f for f in os.listdir(masks_dir) if base_name in f]
            if potential_masks:
                mask_path = os.path.join(masks_dir, potential_masks[0])
            else:
                continue

        image = cv2.imread(img_path)
        if image is None: continue
        
        mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gt is None: continue
        
        # Resize masca la dimensiunea imaginii dacă e necesar
        if mask_gt.shape[:2] != image.shape[:2]:
            mask_gt = cv2.resize(mask_gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # A. Detecție cu YOLO pentru Bounding Box
        results_yolo = yolo_model(image, verbose=False)
        
        dice_score = 0
        iou_score = 0
        
        if results_yolo[0].boxes:
            best_box = results_yolo[0].boxes[0] 
            bbox = best_box.xyxy.cpu().numpy()[0] # [x1, y1, x2, y2]
            
            results_sam = sam_model(image, bboxes=[bbox], verbose=False)
            
            if results_sam[0].masks:
                mask_pred = results_sam[0].masks.data[0].cpu().numpy() # Prima mască
                if mask_pred.shape != mask_gt.shape:
                     mask_pred = cv2.resize(mask_pred.astype(np.float32), (mask_gt.shape[1], mask_gt.shape[0]))
                
                dice_score, iou_score = compute_dice_iou(mask_gt, mask_pred)
        
        tumor_size = compute_tumor_size(mask_gt)
        curvature = compute_curvature(mask_gt)

        results.append({
            'Image': img_name,
            'Dice': dice_score,
            'IoU': iou_score,
            'Tumor_Size': tumor_size,
            'Curvature_Complexity': curvature
        })

    if not results:
        print("Nu s-au generat rezultate. Verificați directoarele și potrivirea numelor.")
        return

    df = pd.DataFrame(results)
    
    output_csv = "rezultate_segmentare_sam_tcga.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nRezultatele salvate în {output_csv}")
    
    avg_dice = df['Dice'].mean()
    avg_iou = df['IoU'].mean()
    print(f"\nAverage Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score:  {avg_iou:.4f}")

    plots_dir = "grafice_segmentare"
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure()
    sns.scatterplot(data=df, x='Tumor_Size', y='Dice', hue='Dice', palette='viridis', alpha=0.7)
    plt.title(f"Segmentation Performance vs Tumor Size\nAvg Dice: {avg_dice:.2f}")
    plt.xlabel("Tumor Size (pixels)")
    plt.ylabel("Dice Score")
    plt.savefig(os.path.join(plots_dir, "dice_vs_size.png"))
    plt.close()

    plt.figure()
    sns.scatterplot(data=df, x='Curvature_Complexity', y='Dice', hue='Dice', palette='magma', alpha=0.7)
    plt.title("Segmentation Performance vs Edge Curvature/Complexity")
    plt.xlabel("Shape Complexity (Irregularity)")
    plt.ylabel("Dice Score")
    plt.savefig(os.path.join(plots_dir, "dice_vs_curvature.png"))
    plt.close()


    try:
        df['Size_Bin'] = pd.qcut(df['Tumor_Size'], q=5, labels=False)
        df['Curve_Bin'] = pd.qcut(df['Curvature_Complexity'], q=5, labels=False)
        
        pivot_table = df.pivot_table(values='Dice', index='Curve_Bin', columns='Size_Bin', aggfunc='mean')
        
        plt.figure()
        sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', fmt='.2f')
        plt.title("Heatmap: Avg Dice Score by Tumor Size and Complexity")
        plt.xlabel("Tumor Size Percentile (0=Small, 4=Large)")
        plt.ylabel("Complexity Percentile (0=Simple, 4=Complex)")
        plt.savefig(os.path.join(plots_dir, "dice_heatmap_aggregations.png"))
        plt.close()
    except Exception as e:
        print(f"Nu s-a putut genera heatmap-ul agregat (posibil prea puține date): {e}")

    # E. Histogram
    plt.figure()
    sns.histplot(df['Dice'], bins=20, kde=True, color='blue')
    plt.title("Distribution of Dice Scores (SAM on TCGA)")
    plt.xlabel("Dice Score")
    plt.savefig(os.path.join(plots_dir, "dice_distribution.png"))
    plt.close()

    print(f"\nGraficele au fost salvate în directorul: {os.path.abspath(plots_dir)}")
    print("Script finalizat cu succes.")

if __name__ == "__main__":
    run_evaluation()
