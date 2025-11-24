import os
import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ============================================================
# REZOLVARE BUG: CURĂȚARE CACHE
# ============================================================
tf.keras.backend.clear_session()

# ============================================================
# PATHS
# ============================================================
data_dir = r'D:\disertatie\DataSet\archive'
train_dir = os.path.join(data_dir, 'Training')
val_dir   = os.path.join(data_dir, 'Testing')

MODEL_SAVE_PATH = r'D:\disertatie\Efficient0-7\B6'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
MODEL_FILENAME = os.path.join(MODEL_SAVE_PATH, 'efficientnet_b6_augumentat.keras')

# ============================================================
# PARAMETERS
# ============================================================
IMG_SIZE = 240
BATCH_SIZE = 16
EPOCHS = 5
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ============================================================
# CUSTOM AUGMENTATION
# ============================================================
def custom_augmentation(image):
    img = (image * 255).astype(np.uint8)
    h, w = img.shape[:2]

    # flip orizontal (30%)
    if random.random() < 0.3:
        img = cv2.flip(img, 1)

    # rotire mică (-10° până la +10°)
    angle = random.uniform(-10, 10)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # zoom (0.9–1.1x)
    scale_factor = round(random.uniform(0.9, 1.1), 2)
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    scaled_img = cv2.resize(img, (new_w, new_h))

    if scale_factor > 1:  # zoom in
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        img = scaled_img[start_y:start_y + h, start_x:start_x + w]
    else:  # zoom out
        canvas = np.zeros_like(img)
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = scaled_img
        img = canvas

    # zgomot redus (1%)
    prob = 0.01
    noisy = np.copy(img)
    total_pixels = img.size // 3
    num_salt = int(prob * total_pixels / 2)
    num_pepper = int(prob * total_pixels / 2)

    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = [255, 255, 255]
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = [0, 0, 0]

    img = noisy
    return img.astype(np.float32) / 255.0

# ============================================================
# DATA GENERATORS (forțare RGB)
# ============================================================
train_datagen = ImageDataGenerator(preprocessing_function=custom_augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'  # foarte important!
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)

# ============================================================
# BUILD MODEL (forțăm 3 canale)
# ============================================================
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = EfficientNetB6(
    include_top=False,
    input_tensor=inputs,
    pooling='avg',
    weights=None  # folosim pretrainare
)
base_model.trainable = True

x = base_model(inputs, training=True)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(CLASSES), activation='softmax')(x)
model = models.Model(inputs, outputs)

# ============================================================
# COMPILE MODEL
# ============================================================
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ============================================================
# CALLBACKS
# ============================================================
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)

# ============================================================
# TRAIN MODEL
# ============================================================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# ============================================================
# SAVE MODEL
# ============================================================
model.save(MODEL_FILENAME)
print(f"✅ Model salvat cu succes în: {MODEL_FILENAME}")

# ============================================================
# PLOT TRAINING
# ============================================================
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Evoluția acurateței - EfficientNetB4 (augmentare echilibrată)')
plt.xlabel('Epoci')
plt.ylabel('Acuratețe')
plt.legend()
plt.grid(True)
plt.show()
