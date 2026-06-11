import os
import numpy as np
import tensorflow as tf
from PIL import Image

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BRAIN_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "classification_resnet50.keras")

# =========================
# LOAD MODEL
# =========================
brain_model = tf.keras.models.load_model(BRAIN_MODEL_PATH, compile=False)

# =========================
# CLASS LABELS (IMPORTANT ⚠️)
# CHANGE ONLY AFTER YOU VERIFY training order
# =========================
CLASS_NAMES = ["CLASS_0", "CLASS_1"]

# =========================
# PREPROCESS
# =========================
def preprocess(image):
    image = image.convert("RGB")

    target_size = brain_model.input_shape[1:3]
    image = image.resize(target_size)

    image = np.array(image).astype(np.float32)

    # SAFE normalization (works even if training differs slightly)
    image = image / 255.0

    return np.expand_dims(image, axis=0)

# =========================
# PREDICTION FUNCTION
# =========================
def predict_brain(image):

    img = preprocess(image)
    pred = brain_model.predict(img, verbose=0)[0]

    # DEBUG OUTPUT (VERY IMPORTANT)
    print("RAW PREDICTION:", pred)

    # =========================
    # CASE 1: Softmax model (2 outputs)
    # =========================
    if len(pred) == 2:
        idx = int(np.argmax(pred))
        confidence = float(pred[idx])

        label = CLASS_NAMES[idx]

        return label, confidence

    # =========================
    # CASE 2: Sigmoid model (1 output)
    # =========================
    elif len(pred) == 1:
        prob = float(pred[0])

        if prob >= 0.5:
            return "NOT BRAIN IMAGE", prob
        else:
            return "BRAIN DETECTED", 1 - prob

    # =========================
    # FALLBACK (never crash)
    # =========================
    else:
        return "UNKNOWN MODEL OUTPUT", 0.0


