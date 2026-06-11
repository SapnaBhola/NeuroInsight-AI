import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import streamlit as st
from PIL import Image
import numpy as np
import brain
import tumor
import segment

import random
from report_generator import generate_ai_report
# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="NeuroInsight Explainable AI", layout="wide")

st.title(" NeuroInsight Explainable AI")
st.subheader("Explainable Brain Tumor Detection & Analysis System")
st.markdown("---")

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title(" Brain Tumor Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Brain Tumor",
        "Causes of Tumor",
        "Types of Tumors",
        "Common Types",
        "Report"
    ]
)

uploaded_file = None  # FIX: global scope

# =========================
# 1. BRAIN TUMOR
# =========================
if page == "Brain Tumor":
    st.title("Brain Tumor Overview")
    st.markdown("""
    A **brain tumor** is an abnormal growth of cells in the brain.

    It can affect brain function depending on size and location.
    """)

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["jpg", "png", "jpeg"],
        key="main_uploader"
    )

# =========================
# 2. CAUSES
# =========================
elif page == "Causes of Tumor":
    st.title("Causes of Brain Tumor")
    st.markdown("""
    - Genetic mutations
    - Radiation exposure
    - Family history
    - Environmental factors
    """)

# =========================
# 3. TYPES
# =========================
elif page == "Types of Tumors":
    st.title(" Types of Brain Tumors")
    st.markdown("""
    **Benign Tumors**
    - Slow growing
    - Less harmful

    **Malignant Tumors**
    - Fast growing
    - Can spread aggressively
    """)

# =========================
# 4. COMMON TYPES
# =========================
elif page == "Common Types":
    st.title(" Common Brain Tumor Types")
    st.markdown("""
    - Glioma
    - Meningioma
    - Pituitary Tumor
    - Medulloblastoma
    """)

# =========================
# 5. REPORT
# =========================
elif page == "Report":
    st.title(" Full System Report")

    report_text = """
    BRAIN TUMOR AI SYSTEM REPORT

    1. Overview:
    Brain tumor detection system using AI models.

    2. Causes:
    - Genetic mutations
    - Radiation exposure
    - Family history

    3. Types:
    - Benign
    - Malignant

    4. Common Types:
    - Glioma
    - Meningioma
    - Pituitary Tumor

    5. System Output:
    Includes MRI image analysis, segmentation, and prediction results.
    """

    st.text_area("Report Content", report_text, height=300)

st.text(report)

# =========================
# FULL DOWNLOAD REPORT BUTTON
# =========================

full_report = generate_ai_report(
    brain_label,
    brain_conf,
    tumor_class,
    tumor_conf,
    result
)

st.download_button(
    label="📥 Download Full Detailed Report",
    data=full_report,
    file_name="NeuroInsight_AI_Report.txt",
    mime="text/plain"
)

# =========================
# PROCESS IMAGE (FIXED SCOPE)
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Input Image")
    st.image(image, width=300)


    # =========================
    # STEP 1: BRAIN CHECK
    # =========================
    st.header("Step 1: Brain Check")

    brain_label, brain_conf = brain.predict_brain(image)

    col1, col2 = st.columns(2)
    col1.metric("Result", brain_label)
    col2.metric("Confidence", f"{brain_conf:.4f}")

    if "NOT BRAIN" in brain_label:
        st.error(
            "❌ Invalid image detected.\n\n"
            "This application only works with Brain MRI images.\n"
            "Please upload a valid brain scan to continue."
        )
        st.stop()

    # =========================
    # STEP 2: TUMOR DETECTION
    # =========================
    st.header("Step 2: Tumor Detection")

    tumor_class, tumor_conf, pred, cam_img = tumor.predict_with_gradcam(image)

    col1, col2 = st.columns(2)
    col1.metric("Tumor Type", tumor_class)
    col2.metric("Confidence", f"{tumor_conf:.4f}")

    # CALIBRATION (NEW ADDITION)
    # =========================
    try:
        raw_conf, calibrated_conf, uncertainty = get_calibrated_confidence(pred, T=2.5)
    except:
        raw_conf = tumor_conf
        calibrated_conf = tumor_conf
        uncertainty = 1 - tumor_conf

    col1, col2 = st.columns(2)
    col1.metric("Tumor Type", tumor_class)
    col2.metric("Raw Confidence", f"{raw_conf:.2f}")

    col1, col2 = st.columns(2)
    col1.metric("Calibrated Confidence", f"{calibrated_conf:.2f}")
    col2.metric("Uncertainty", f"{uncertainty:.2f}")


    # Layout: Original Image + Grad-CAM++
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original MRI")
        st.image(image, width=280)

    with col2:
        st.subheader("Tumor Attention (Grad-CAM++)")
        st.image(cam_img.astype("uint8"), width=280)


    # =========================
    # =========================
    # MODEL METRICS (IMAGE-BASED)
    # =========================
    with st.expander("Model Metrics (Image-Based)"):

        # Convert prediction probabilities into meaningful "real-time metrics"
        tumor_prob = float(np.max(pred))  # highest class probability
        no_tumor_prob = float(np.min(pred))  # lower probability

        # Dynamic confidence interpretation
        confidence = tumor_conf

        # Derived image-based metrics
        prediction_strength = "High" if confidence > 0.85 else "Medium" if confidence > 0.6 else "Low"

        uncertainty = 1 - confidence

        # Display
        col1, col2 = st.columns(2)

        col1.metric("Tumor Probability", f"{tumor_prob:.2f}")
        col2.metric("No Tumor Probability", f"{no_tumor_prob:.2f}")

        col3, col4 = st.columns(2)

        col3.metric("Prediction Confidence", f"{confidence:.2f}")
        col4.metric("Uncertainty", f"{uncertainty:.2f}")

        st.write(f"**Prediction Strength:** {prediction_strength}")

        # Visual interpretation bar
        st.progress(confidence)

        st.caption("Image-based metrics derived from model output probabilities (not dataset evaluation)")
        st.info("These metrics reflect THIS MRI scan only, not global training performance")

    # =========================
    # STEP 3: SEGMENTATION
    # =========================
    if tumor_class != "No Tumor":

        st.header("Step 3: Tumor Segmentation")

        original, mask, overlay, result, report = segment.run_segmentation(image)

        # Fix image formats
        original_display = original.astype("uint8")
        mask_display = (mask * 255).astype("uint8")
        overlay_display = overlay.astype("uint8")

        # =========================
        # IMAGE DISPLAY (CLEAN GRID)
        # =========================
        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            st.image(original_display, caption="MRI", width=220)

        with col2:
            st.image(mask_display, caption="Mask", width=220)

        with col3:
            st.image(overlay_display, caption="Overlay", width=220)

        st.markdown("<br>", unsafe_allow_html=True)

        # =========================
        # ANALYSIS
        # =========================
        st.subheader("Tumor Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Tumor %:** {result['tumor_percent']:.2f}%")
            st.write(f"**Area:** {int(result['tumor_area'])} pixels")

            # ✅ NEW: Actionable Severity Logic (ADDED ONLY)
            percent = result['tumor_percent']

            if percent < 2:
                action_severity = "Low Risk (Monitor)"
            elif percent < 10:
                action_severity = "Moderate Risk (Consult Specialist)"
            else:
                action_severity = "High Risk (Urgent Attention)"

            st.write(f"**AI Recommendation:** {action_severity}")

            loc = result["location"]
            if loc:
                st.write(f"**Location:** (x={loc[0]:.1f}, y={loc[1]:.1f})")
            else:
                st.write("**Location:** Not detected")

        with col2:
            st.write(f"**Bounding Box:** {result['bbox']}")
            st.write(f"**Shape:** {result['shape']}")
            st.write(f"**Severity (Model):** {result['severity']}")

        # =========================
        # REPORT
        # =========================
        st.subheader("AI Radiology Report")
        st.text(report)

    else:
        st.success(
          " No tumor detected.\n\n"
          "• The model did not identify any abnormal region.\n"
          "• Segmentation is skipped as there is no tumor present.\n"
          "• No further analysis required."
      )