import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from ultralytics import YOLO, SAM
from tqdm import tqdm

# --- Configurare Plotting ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 7)

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

def compute_tumor_size(mask):
    return np.sum(mask > 0)

def compute_curvature(mask):
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area == 0: return 0
    return (perimeter ** 2) / (4 * np.pi * area)

def compute_dice_iou(mask_true, mask_pred):
    """Calculează Dice și IoU între masca reală și cea prezisă."""
    # Binarizare
    mask_true = (mask_true > 0).astype(np.uint8)
    mask_pred = (mask_pred > 0).astype(np.uint8)

    intersection = np.sum(mask_true * mask_pred)
    sum_true = np.sum(mask_true)
    sum_pred = np.sum(mask_pred)

    dice = (2. * intersection) / (sum_true + sum_pred + 1e-6)
    iou = intersection / (sum_true + sum_pred - intersection + 1e-6)
    
    return dice, iou

def generate_box_plots(df, output_dir):
    """Generează graficele stil Box Plot solicitate, pe baza datelor din CSV."""
    if df.empty:
        print("Date insuficiente pentru generare grafice.")
        return

    # 1. Categorisire Curbură/Complexitate (Low, Medium, High)
    print("\nGenerare categorii de complexitate (Curvature Levels)...")
    try:
        df['Curvature_Level'] = pd.qcut(df['Predicted_Complexity'], q=3, labels=["Low", "Medium", "High"])
    except ValueError:
        print("Avertisment: Nu s-au putut calcula tertilele (qcut). Se folosește o împărțire simplă.")
        df['Curvature_Level'] = pd.cut(df['Predicted_Complexity'], bins=3, labels=["Low", "Medium", "High"])

    df_plot = df[df['Predicted_Tumor_Size'] > 0].copy()

    # --- Ploturi Morfologice (Existente) ---
    plt.figure()
    sns.boxplot(data=df_plot, x='Curvature_Level', y='Predicted_Tumor_Size', hue='Curvature_Level', palette="Set3", legend=False)
    plt.title("Analiză Morfologică: Dimensiune Tumoare vs. Complexitate Formă")
    plt.xlabel("Nivel Complexitate")
    plt.ylabel("Dimensiune (Pixeli)")
    plt.savefig(os.path.join(output_dir, "boxplot_size_vs_curvature.png"))
    plt.close()

    plt.figure()
    sns.regplot(data=df_plot, x='Predicted_Complexity', y='Predicted_Tumor_Size', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title("Corelație: Complexitate vs. Dimensiune")
    plt.xlabel("Complexitate")
    plt.ylabel("Dimensiune")
    plt.savefig(os.path.join(output_dir, "scatter_size_complexity.png"))
    plt.close()

    plt.figure()
    plt.hist2d(df_plot['Predicted_Complexity'], df_plot['Predicted_Tumor_Size'], bins=(30, 30), cmap='Blues')
    plt.colorbar(label='Număr Tumori')
    plt.title("Heatmap Densitate: Distribuția Tumorilor")
    plt.savefig(os.path.join(output_dir, "heatmap_density_distribution.png"))
    plt.close()

    # --- Ploturi Performanță (Dacă există Dice/Confidență) ---
    
    # Confidență
    if 'Confidence' in df_plot.columns and df_plot['Confidence'].sum() > 0:
        plt.figure()
        sns.boxplot(data=df_plot, x='Curvature_Level', y='Confidence', hue='Curvature_Level', palette="Set2", legend=False)
        plt.title("Performanță Detecție: Confidență vs. Complexitate")
        plt.xlabel("Nivel Complexitate")
        plt.ylabel("Confidență")
        plt.savefig(os.path.join(output_dir, "boxplot_confidence_vs_curvature.png"))
        plt.close()

    # Dice Score (Cel cerut de user)
    if 'Dice' in df_plot.columns and df_plot['Dice'].sum() > 0:
        print("Generare grafic Dice Score vs Curvature...")
        plt.figure()
        # Graficul EXACT cerut de user (Box Plot Dice pe nivele de curbură)
        sns.boxplot(data=df_plot, x='Curvature_Level', y='Dice', hue='Curvature_Level', palette="viridis", legend=False)
        plt.title("Performanță Segmentare (Dice Score) vs. Complexitate")
        plt.xlabel("Nivel Complexitate (Curvature Level)")
        plt.ylabel("Dice Score")
        plt.ylim(0, 1.05)
        plt.savefig(os.path.join(output_dir, "boxplot_dice_vs_curvature.png"))
        plt.close()
        print(f"GRAFIC DICE GENERAT: {os.path.join(output_dir, 'boxplot_dice_vs_curvature.png')}")

        # Dice vs Size
        plt.figure()
        sns.scatterplot(data=df_plot, x='Predicted_Tumor_Size', y='Dice', hue='Curvature_Level', palette='viridis')
        plt.title("Dice Score vs. Tumor Size")
        plt.xlabel("Dimensiune")
        plt.ylabel("Dice Score")
        plt.savefig(os.path.join(output_dir, "scatter_dice_vs_size.png"))
        plt.close()

def run_evaluation():
    print("=== Script Segmentare Avansată (Cu suport Dice Score) ===")
    print("1. Rulează Evaluare Nouă")
    print("2. Generează Grafice din CSV Existent")
    
    choice = input("\nAlegeți opțiunea (1 sau 2): ").strip()

    if choice == '2':
        default_csv = r"D:\Disertatie\BE-disertatie\djangoProject1\rezultate_analiza_segmentare.csv"
        csv_path = default_csv if os.path.exists(default_csv) else select_file("Selectați CSV", [("CSV Files", "*.csv")])
        if not csv_path: return
        
        df = pd.read_csv(csv_path)
        output_dir = os.path.join(os.path.dirname(csv_path), "grafice_recreate")
        os.makedirs(output_dir, exist_ok=True)
        generate_box_plots(df, output_dir)
        print("\nGenerare completă!")
        return

    # Mod Evaluare Nouă
    yolo_path = select_file("Model YOLO (.pt)", [("YOLO Model", "*.pt")])
    if not yolo_path: return
    
    sam_path = select_file("Model SAM (.pt)", [("SAM Model", "*.pt")])
    if not sam_path: return
    
    images_dir = select_directory("Director Imagini Test")
    if not images_dir: return

    # OPȚIONAL: Director Măști
    use_masks = False
    masks_dir = None
    print("\n[Opțional] Selectați directorul cu Măști Ground Truth pentru calcul Dice Score.")
    print("Dacă nu aveți măști, închideți fereastra de dialog sau dați Cancel.")
    masks_dir = select_directory("Director Măști (Opțional)")
    
    if masks_dir and os.path.exists(masks_dir):
        use_masks = True
        print(f"Măști selectate: {masks_dir} -> CALCUL DICE ACTIVAT")
    else:
        print("Director măști nespecificat -> CALCUL DICE DEZACTIVAT (Doar analiză morfologică)")

    output_vis_dir = os.path.join(os.path.dirname(__file__), "vizualizare_segmentare_nou")
    os.makedirs(output_vis_dir, exist_ok=True)
    
    yolo_model = YOLO(yolo_path)
    sam_model = SAM(sam_path)
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results = []

    print(f"Procesare {len(image_files)} imagini...")
    for img_name in tqdm(image_files):
        img_path = os.path.join(images_dir, img_name)
        image = cv2.imread(img_path)
        if image is None: continue
        
        # Gestionare mască (dacă există)
        mask_gt = None
        if use_masks:
            # Încercăm să găsim masca (nume identic sau similar)
            mask_path = os.path.join(masks_dir, img_name)
            if not os.path.exists(mask_path):
                 # Fallback: poate e png în loc de jpg
                 base_name = os.path.splitext(img_name)[0]
                 for ext in ['.png', '.jpg', '.tif', '_mask.png']:
                     if os.path.exists(os.path.join(masks_dir, base_name + ext)):
                         mask_path = os.path.join(masks_dir, base_name + ext)
                         break
            
            if os.path.exists(mask_path):
                mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_gt is not None:
                     if mask_gt.shape[:2] != image.shape[:2]:
                        mask_gt = cv2.resize(mask_gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Detecție & Segmentare
        res_yolo = yolo_model(image, verbose=False)
        
        has_det, size, comp, conf, dice, iou = False, 0, 0, 0, 0, 0
        
        if res_yolo[0].boxes:
            best_box = res_yolo[0].boxes[0]
            bbox = best_box.xyxy.cpu().numpy()[0]
            conf = float(best_box.conf)
            
            res_sam = sam_model(image, bboxes=[bbox], verbose=False)
            if res_sam[0].masks:
                mask = res_sam[0].masks.data[0].cpu().numpy()
                mask_u8 = (mask * 255).astype(np.uint8)
                if mask_u8.shape[:2] != image.shape[:2]:
                    mask_u8 = cv2.resize(mask_u8, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                size = compute_tumor_size(mask_u8)
                comp = compute_curvature(mask_u8)
                has_det = True
                
                # Calcul Dice dacă avem mască
                if mask_gt is not None:
                    dice, iou = compute_dice_iou(mask_gt, mask_u8)
                
                # ... salvare overlay cod existent ...

        results.append({
            'Image': img_name,
            'Detected': has_det,
            'Predicted_Tumor_Size': size,
            'Predicted_Complexity': comp,
            'Confidence': conf,
            'Dice': dice, # Va fi 0 dacă nu avem mască
            'IoU': iou
        })

    df = pd.DataFrame(results)
    out_csv = "rezultate_segmentare_nou.csv"
    df.to_csv(out_csv, index=False)
    print(f"Salvat rezultate noi în: {out_csv}")
    
    generate_box_plots(df, "grafice_segmentare_nou")

if __name__ == "__main__":
    run_evaluation()
