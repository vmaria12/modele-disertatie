import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras import layers, models

# -------------------------
# PATHS
data_dir = r'/DataSet/archive'
train_dir = os.path.join(data_dir, 'Training')
val_dir   = os.path.join(data_dir, 'Testing')

MODEL_SAVE_PATH = r'/Models/Yolo_v8'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
MODEL_FILENAME = os.path.join(MODEL_SAVE_PATH, 'efficientnet_multi_class.h5')

# -------------------------
# PARAMETERS
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 3
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# -------------------------
# DATA GENERATORS
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
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
base_model = EfficientNetB5(include_top=False, input_tensor=inputs, pooling='avg', weights='imagenet')
base_model.trainable = True

x = base_model(inputs, training=True)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(len(CLASSES), activation='softmax')(x)
model = models.Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------
# TRAIN MODEL
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# -------------------------
# SAVE MODEL
model.save(MODEL_FILENAME)
print(f"Model salvat în: {MODEL_FILENAME}")

# -------------------------
# GRAD-CAM FUNCTION
def get_gradcam(model, img_array, class_index=None, last_conv_layer_name='top_conv'):
    """
    img_array: numpy array, shape=(1, IMG_SIZE, IMG_SIZE, 3)
    class_index: index-ul clasei pentru care facem Grad-CAM; dacă None => cea mai probabilă clasă
    """
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(last_conv_layer_name).output, model.output])
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
# FUNCTION TO DISPLAY GRAD-CAM
def display_gradcam(img_path, model):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)/255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    # Grad-CAM
    heatmap = get_gradcam(model, img_array_exp)
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.uint8(img_array*255), 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(6,6))
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

# -------------------------
# EXAMPLE USAGE
# display_gradcam(r'D:\Disertatie\Brain\demo\DataSet\archive\Testing\glioma\img_example.jpg', model)
