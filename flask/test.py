# =========================
# IMPORTS FIRST
# =========================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cv2
import warnings
import tensorflow as tf

# =========================
# SAFE ENV SETTINGS
# =========================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore")

print("LOADING SINGLE MODEL\n")

# =========================
# MODEL PATH
# =========================
MODEL_PATH = r"D:\NeuroInsight\models\brain_tumor_segmentation.keras"

print("Checking path:", MODEL_PATH)

# =========================
# CHECK MODEL EXISTS
# =========================
if not os.path.exists(MODEL_PATH):
    print(" Model NOT FOUND!")
    input("Press Enter to exit...")
    exit()

print("Model FOUND")

# =========================
# LOAD MODEL
# =========================
print("Loading model...")

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )

    print("✅ Model loaded successfully!")

except Exception as e:
    print("\n ERROR WHILE LOADING MODEL")
    print(str(e))
    input("\nPress Enter to exit...")
    exit()
