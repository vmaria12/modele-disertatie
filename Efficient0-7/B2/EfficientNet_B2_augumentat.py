import os
import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# -------------------------
# PATHS
data_dir = r'D:\disertatie\DataSet\archive'
train_dir = os.path.join(data_dir, 'Training')
val_dir   = os.path.join(data_dir, 'Testing')

MODEL_SAVE_PATH = r'D:\disertatie\Efficient0-7\B2'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

MODEL_FILENAME = os.path.join(MODEL_SAVE_PATH, 'efficientnet_b2_augumentat.keras')

# -------------------------
# PARAMETERS
IMG_SIZE = 240   # recomandat pentru EfficientNetV2B1 (default input 240x240)
BATCH_SIZE = 16
EPOCHS = 5
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# -------------------------
# CUSTOM AUGMENTATION FUNCTION
def custom_augmentation(image):
    img = (image * 255).astype(np.uint8)
    h, w = img.shape[:2]

    # 1. Flip orizontal (30%)
    if random.random() < 0.3:
        img = cv2.flip(img, 1)

    # 2. Rotire mică (-10° până la +10°)
    angle = random.uniform(-10, 10)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # 3. Zoom moderat (0.9–1.1x)
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

    # 4. Zgomot foarte redus (1%)
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


# -------------------------
# DATA GENERATORS
train_datagen = ImageDataGenerator(preprocessing_function=custom_augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# -------------------------
# BUILD MODEL
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = EfficientNetV2B2(include_top=False, input_tensor=inputs, pooling='avg', weights='imagenet')
base_model.trainable = True  # fine-tuning activ

x = base_model(inputs, training=True)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(CLASSES), activation='softmax')(x)
model = models.Model(inputs, outputs)

# -------------------------
# COMPILE MODEL
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------
# CALLBACKS
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)

# -------------------------
# TRAIN MODEL
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# -------------------------
# SAVE MODEL (.keras format)
model.save(MODEL_FILENAME)
print(f"✅ Model salvat cu succes în: {MODEL_FILENAME}")

# -------------------------
# PLOT TRAINING RESULTS
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Evoluția acurateței - EfficientNetV2B1 (augmentare echilibrată)')
plt.xlabel('Epoci')
plt.ylabel('Acuratețe')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# GRAD-CAM FUNCTION
def get_gradcam(model, img_array, class_index=None, last_conv_layer_name='top_conv'):
    """
    img_array: numpy array, shape=(1, IMG_SIZE, IMG_SIZE, 3)
    class_index: index-ul clasei pentru care facem Grad-CAM; dacă None => cea mai probabilă clasă
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap.numpy(), (IMG_SIZE, IMG_SIZE))
    return heatmap

# -------------------------
# DISPLAY GRAD-CAM
def display_gradcam(img_path, model):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    heatmap = get_gradcam(model, img_array_exp)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.uint8(img_array * 255), 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(6, 6))
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

# -------------------------
# EXAMPLE USAGE (optional)
# display_gradcam(r'D:\Disertatie\Brain\demo\DataSet\archive\Testing\glioma\img_example.jpg', model)
