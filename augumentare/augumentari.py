import cv2
import numpy as np
import random
from tkinter import filedialog, Tk
import matplotlib.pyplot as plt

def procesare_imagine():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Selectează o imagine",
        filetypes=[("Fișiere imagine", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )

    if not file_path:
        print("Nu s-a selectat nicio imagine.")
        return

    img = cv2.imread(file_path)
    if img is None:
        print("Eroare la încărcarea imaginii.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # ---- 1. Oglindire orizontală ----
    flip_oriz = cv2.flip(img_rgb, 1)

    # ---- 2. Oglindire verticală ----
    flip_vert = cv2.flip(img_rgb, 0)

    # ---- 3. Rotire aleatorie ----
    angle = random.randint(-0, 180)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_rgb, M, (w, h))

    # ---- 4. Scalare (zoom in / zoom out) ----
    scale_factor = round(random.uniform(1, 1.2), 2)
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    scaled_img = cv2.resize(img_rgb, (new_w, new_h))

    if scale_factor > 1:
        # zoom in → decupăm centrul la dimensiunea originală
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        scaled = scaled_img[start_y:start_y + h, start_x:start_x + w]
    else:
        # zoom out → adăugăm margini negre pentru a păstra dimensiunea
        canvas = np.zeros_like(img_rgb)
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = scaled_img
        scaled = canvas

    # ---- 5. Zgomot "sare și piper" ----
    def salt_and_pepper(image, prob=0.2):
        noisy = np.copy(image)
        total_pixels = image.size // 3
        num_salt = int(prob * total_pixels / 2)
        num_pepper = int(prob * total_pixels / 2)

        # sare (pixeli albi)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = [255, 255, 255]

        # piper (pixeli negri)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = [0, 0, 0]
        return noisy

    noisy = salt_and_pepper(img_rgb)

    # ---- Afișare ----
    titles = [
        "Imagine Originală",
        "Flip Orizontal",
        "Flip Vertical",
        f"Rotire ({angle}°)",
        f"Zoom ({scale_factor}x)",
        "Zgomot sare și piper"
    ]
    images = [img_rgb, flip_oriz, flip_vert, rotated, scaled, noisy]

    plt.figure(figsize=(15, 8))
    for i, (title, image) in enumerate(zip(titles, images)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Apelăm funcția
procesare_imagine()
