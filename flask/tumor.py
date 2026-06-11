import os  # ✅ FIX ADDED

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "final_resnet_model.keras")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
input_shape = model.input_shape[1:3]

class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]

def preprocess(image):
    image = image.convert("RGB")
    image = image.resize(input_shape)  # safer
    image = np.array(image)

    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = preprocess_input(image.astype(np.float32))
    return np.expand_dims(image, axis=0)

def predict_tumor(image):
    processed = preprocess(image)
    predictions = model.predict(processed)[0]

    idx = np.argmax(predictions)
    class_name = class_names[idx]
    confidence = float(predictions[idx])

    return class_name, confidence, predictions, processed

def make_gradcam_plus_plus_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape(persistent=True) as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(1, 2))

    conv_outputs = conv_outputs[0]
    weights = weights[0]

    heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

def overlay(heatmap, img):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

def predict_with_gradcam(image):

    class_name, confidence, predictions, processed = predict_tumor(image)

    last_conv_layer = None
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                last_conv_layer = layer.name
                break
        except:
            continue

    if last_conv_layer is None:  # ✅ FIX
        raise ValueError("No convolutional layer found")

    heatmap = make_gradcam_plus_plus_heatmap(processed, model, last_conv_layer)

    img = np.array(image.resize(input_shape))
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    cam_img = overlay(heatmap, img)

    return class_name, confidence, predictions, cam_img