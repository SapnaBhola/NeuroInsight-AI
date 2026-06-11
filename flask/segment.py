import os  # ✅ FIX

import numpy as np
import cv2
import tensorflow as tf
from skimage.measure import label, regionprops

IMG_SIZE = 256

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "brain_tumor_segmentation.keras")

# ✅ FIX: Lazy loading
model = None

def get_model():
    global model
    if model is None:
        print("🧠 Loading segmentation model...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def tumor_area(mask):
    area = int(np.sum(mask))
    percent = (area / mask.size) * 100
    return area, percent


def tumor_location(mask):
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    return (float(np.mean(coords[1])), float(np.mean(coords[0])))


def tumor_bbox(mask):
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    return (
        int(coords[1].max() - coords[1].min()),
        int(coords[0].max() - coords[0].min())
    )


def tumor_shape_features(mask):
    labeled = label(mask)
    props = regionprops(labeled)
    if len(props) == 0:
        return None
    r = props[0]
    return {
        "eccentricity": float(r.eccentricity),
        "solidity": float(r.solidity)
    }


def estimate_severity(percent):
    if percent < 2:
        return "Very Small"
    elif percent < 10:
        return "Mild"
    elif percent < 25:
        return "Moderate"
    else:
        return "Large / Severe"


def generate_report(result):

    percent = result["tumor_percent"]
    area = result["tumor_area"]
    bbox = result["bbox"]
    shape = result["shape"]
    severity = result["severity"]
    location = result["location"]

    # Smart descriptions
    size_desc = "small" if percent < 2 else "moderate" if percent < 10 else "large"

    if shape and shape.get("eccentricity", 0) > 0.8:
        shape_desc = "irregular"
    else:
        shape_desc = "well-defined"

    comp_desc = "solid mass"
    spread_desc = "localized" if percent < 10 else "diffuse"

    if location:
        loc_desc = f"located near (x={location[0]:.1f}, y={location[1]:.1f})"
    else:
        loc_desc = "location unclear"

    return f"""
RADIOLOGY REPORT (AI GENERATED)

Findings:
A {size_desc} lesion is observed, occupying approximately {percent:.2f}% of the scanned brain region.

The lesion appears {shape_desc}, suggesting a {comp_desc}. It is {spread_desc} and is {loc_desc}.

The bounding dimensions are {bbox}, with total area {area} pixels.

Impression:
Overall tumor burden is classified as {severity.lower()}.

Note:
AI-generated report — not a clinical diagnosis.
"""

def run_segmentation(image):

    img_np = np.array(image)

    if len(img_np.shape) == 3:
        original = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        original = img_np

    img = preprocess_image(original)
    input_img = np.expand_dims(img, axis=0)

    model = get_model()  # ✅ FIX

    pred = model.predict(input_img, verbose=0)[0]

    # ✅ SAFE MASK
    if len(pred.shape) == 3:
        pred_mask = pred[:, :, 0]
    else:
        pred_mask = pred

    mask = (pred_mask > 0.5).astype(np.uint8)

    area, percent = tumor_area(mask)
    location = tumor_location(mask)
    bbox = tumor_bbox(mask)
    shape = tumor_shape_features(mask)
    severity = estimate_severity(percent)

    result = {
        "tumor_percent": percent,
        "tumor_area": area,
        "location": location,
        "bbox": bbox,
        "shape": shape,
        "severity": severity
    }

    original_resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    original_bgr = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2BGR)

    colored_mask = np.zeros_like(original_bgr)
    colored_mask[:, :, 2] = mask * 255

    overlay = cv2.addWeighted(original_bgr, 0.75, colored_mask, 0.35, 0)

    return original_resized, mask, overlay, result, generate_report(result)