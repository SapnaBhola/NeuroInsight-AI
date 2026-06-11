# NeuroInsight AI

## NeuroInsight AI: Brain Tumor Detection, Classification, Localization and Segmentation Using Deep Learning

### 6-Month Industrial Training Project

# Introduction

NeuroInsight AI is an intelligent medical imaging platform developed for automated Brain MRI analysis using Deep Learning, Computer Vision, and Explainable Artificial Intelligence (XAI).

The system assists in detecting and classifying brain tumors from MRI scans, highlighting the regions responsible for predictions using Grad-CAM, performing tumor segmentation using a U-Net architecture, and generating automated AI-assisted radiology reports.

The objective of this project is to provide a fast, efficient, and explainable preliminary analysis of Brain MRI scans while demonstrating the practical application of Artificial Intelligence in healthcare.

---

# Problem Statement

Brain tumor diagnosis traditionally requires expert radiologists to manually examine MRI scans, which can be time-consuming and subject to human error.

Challenges include:

- Large volume of MRI data
- Time-consuming manual analysis
- Difficulty in tumor localization
- Need for explainable predictions
- Requirement for accurate tumor segmentation

NeuroInsight AI addresses these challenges by automating the MRI analysis workflow using Deep Learning and Computer Vision techniques.

---

# Project Objectives

The primary objectives of NeuroInsight AI are:

- Validate uploaded Brain MRI scans.
- Detect the presence of brain tumors.
- Classify tumor types using Deep Learning.
- Visualize prediction regions through Explainable AI.
- Segment tumor regions for detailed analysis.
- Generate automated radiology-style reports.
- Export reports in PDF format.
- Maintain analysis history for future reference.

---

# Key Features

## Brain MRI Validation

The system verifies whether the uploaded image is a valid Brain MRI scan before performing analysis.

### Capabilities

- Brain MRI verification
- Invalid image rejection
- Prevention of incorrect analysis

---

## Brain Tumor Classification

The classification module predicts the tumor category using a Deep Learning model.

### Supported Classes

| Class | Description |
|---------|-------------|
| Glioma | Tumor originating from glial cells |
| Meningioma | Tumor arising from the meninges |
| Pituitary | Tumor affecting the pituitary gland |
| No Tumor | MRI scan without detectable tumor |

---

## Explainable AI (Grad-CAM)

Grad-CAM visualization highlights the MRI regions that influenced the model prediction.

### Benefits

- Improved model transparency
- Better prediction interpretability
- Explainable AI support

---

## Tumor Segmentation

The segmentation module extracts and isolates tumor regions from MRI scans.

### Outputs

- Tumor mask generation
- Tumor localization
- Tumor boundary extraction
- Tumor area estimation

---

## AI-Powered Report Generation

The platform automatically generates radiology-style reports containing:

- Predicted diagnosis
- Confidence score
- Tumor classification
- Clinical findings
- Recommendations

---

## PDF Report Export

Users can download detailed reports in PDF format for:

- Documentation
- Record keeping
- Clinical review
- Academic evaluation

---

## User Authentication

The platform provides:

- User registration
- Secure login
- Session management

---

## History Management

The system stores previous analyses and reports for future reference.

---

## Contact Support Module

Integrated support system with:

- Contact form submissions
- Email notifications
- Message management

---

# Technology Stack

| Category | Technology |
|------------|------------|
| Frontend | HTML5, CSS3, JavaScript, Bootstrap |
| Template Engine | Jinja2 |
| Backend | Python, Flask |
| Deep Learning Framework | TensorFlow, Keras |
| Classification Model | ResNet50 |
| Segmentation Model | U-Net |
| Explainable AI | Grad-CAM |
| Computer Vision | OpenCV, Pillow |
| Data Processing | NumPy |
| Data Storage | JSON |
| Report Generation | ReportLab |
| Email Services | Flask-Mail, Gmail SMTP |
| Containerization | Docker |
| Version Control | Git, GitHub |
| Development Environment | Visual Studio Code |

---

# Deep Learning Models

## Tumor Classification Model

### Architecture

ResNet50 (Transfer Learning)

### Purpose

- Brain Tumor Classification
- Feature Extraction
- Multi-Class Prediction

### Output Classes

- Glioma
- Meningioma
- Pituitary
- No Tumor

---

## Tumor Segmentation Model

### Architecture

U-Net Segmentation Network

### Purpose

- Tumor Localization
- Tumor Mask Generation
- Tumor Area Estimation

### Advantages

- Biomedical image specialization
- Pixel-level segmentation
- Accurate boundary detection

---

# Dataset Sources

## Brain Tumor Classification Dataset

Dataset Link:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

### Classes

- Glioma
- Meningioma
- Pituitary
- No Tumor

---

## Brain Tumor Segmentation Dataset

Dataset Link:

https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation

### Applications

- Tumor Segmentation
- Tumor Mask Generation
- Tumor Area Analysis

---

# System Workflow

```text
User Uploads MRI Scan
        │
        ▼
Brain MRI Validation
        │
        ▼
ResNet50 Classification
        │
        ▼
Grad-CAM Generation
        │
        ▼
U-Net Segmentation
        │
        ▼
Tumor Analysis
        │
        ▼
AI Report Generation
        │
        ▼
PDF Report Export
        │
        ▼
History Storage
```

---

# How NeuroInsight AI Works

### Step 1

The user uploads a Brain MRI image.

### Step 2

The system validates whether the image is a valid Brain MRI scan.

### Step 3

The ResNet50 classification model predicts the tumor category.

### Step 4

Grad-CAM generates a heatmap showing regions influencing the prediction.

### Step 5

The U-Net model performs tumor segmentation and extracts tumor boundaries.

### Step 6

Tumor analysis is performed to estimate affected regions.

### Step 7

An AI-assisted radiology report is generated.

### Step 8

The report can be downloaded as a PDF document and stored in analysis history.

---

# Installation and Setup

## Clone Repository

```bash
git clone https://github.com/yourusername/NeuroInsight-AI.git

cd NeuroInsight-AI
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
python flask/app.py
```

---

# Running with Docker

## Build Docker Image

```bash
docker build -t neuroinsight-ai .
```

## Run Docker Container

```bash
docker run -p 5000:5000 neuroinsight-ai
```

---

# How To Use

1. Register a new account.
2. Login to the platform.
3. Upload a Brain MRI image.
4. Wait for analysis completion.
5. View classification results.
6. Examine Grad-CAM visualization.
7. Review segmentation output.
8. Generate AI report.
9. Download PDF report.
10. Access previous reports through History.

---

# Future Enhancements

- DICOM File Support
- 3D MRI Visualization
- Cloud Deployment
- Multi-Disease Detection
- Doctor Dashboard
- Patient Management System
- Hospital Information System Integration
- Real-Time Clinical Assistance
- Advanced Explainable AI Features

---

# Medical Disclaimer

This software has been developed solely for educational, research, and industrial training purposes.

The predictions and reports generated by this system should not be considered a substitute for professional medical diagnosis, treatment planning, or clinical decision-making.

Always consult qualified healthcare professionals for medical advice and diagnosis.

---

# Author

**Sapna**

Bachelor of Technology (Information Technology)

Final Year Project



# License

This project is intended for educational, academic, and research purposes.#
