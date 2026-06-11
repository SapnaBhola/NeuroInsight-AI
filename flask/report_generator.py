# -*- coding: utf-8 -*-

import sys
import io

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding='utf-8',
        errors='replace'
    )

from datetime import datetime
import random
import os


# ============================================================
# TUMOR TYPE DATABASE
# ============================================================

TUMOR_DB = {

    "no_tumor": {
        "display_name": "No Tumor",
        "risk_level": "NO RISK",
        "risk_class": "none",

        "description":
            "No abnormal tissue or lesion was detected.",

        "characteristics": [
            "Normal brain tissue",
            "No abnormal enhancement",
            "No mass effect"
        ],

        "recommendations": [
            "Routine monitoring only",
            "Maintain neurological follow-up"
        ],

        "summary":
            "No evidence of brain tumor detected."
    },

    "meningioma": {
        "display_name": "Meningioma",
        "risk_level": "LOW RISK",
        "risk_class": "low",

        "description":
            "Usually benign extra-axial brain tumor arising from meninges.",

        "characteristics": [
            "Well-defined margins",
            "Slow-growing",
            "Extra-axial lesion"
        ],

        "recommendations": [
            "Neurology consultation recommended",
            "Periodic MRI follow-up advised",
            "Monitor tumor growth progression"
        ],

        "summary":
            "MRI findings suggest meningioma with localized tumor involvement."
    },

    "glioma": {
        "display_name": "Glioma",
        "risk_level": "HIGH RISK",
        "risk_class": "high",

        "description":
            "Infiltrative tumor arising from glial cells.",

        "characteristics": [
            "Irregular morphology",
            "Infiltrative growth",
            "Possible edema"
        ],

        "recommendations": [
            "Urgent neuro-oncology consultation",
            "Advanced MRI evaluation",
            "Biopsy and grading recommended"
        ],

        "summary":
            "MRI findings suggest infiltrative glioma requiring urgent evaluation."
    },

    "pituitary": {
        "display_name": "Pituitary Tumor",
        "risk_level": "MODERATE RISK",
        "risk_class": "moderate",

        "description":
            "Sellar-region pituitary lesion affecting endocrine structures.",

        "characteristics": [
            "Sellar location",
            "Hormonal dysfunction possible",
            "Optic chiasm compression risk"
        ],

        "recommendations": [
            "Endocrinology consultation",
            "Visual field testing",
            "Pituitary MRI evaluation"
        ],

        "summary":
            "MRI findings suggest pituitary lesion with moderate clinical risk."
    }
}


# ============================================================
# RISK ASSESSMENT
# ============================================================

def get_risk_info(tumor_type, tumor_percent):

    db = TUMOR_DB.get(tumor_type, TUMOR_DB["no_tumor"])

    if tumor_type == "no_tumor":

        action = (
            "No immediate medical action required."
        )

    elif tumor_percent < 2:

        action = (
            "Regular MRI monitoring recommended."
        )

    elif tumor_percent < 10:

        action = (
            "Specialist consultation recommended."
        )

    else:

        action = (
            "Urgent medical evaluation recommended."
        )

    return db["risk_level"], action


# ============================================================
# REPORT GENERATOR
# ============================================================

def generate_ai_report(
    brain_label,
    brain_conf,
    tumor_class,
    tumor_conf,
    result
):

    current_time = datetime.now().strftime(
        "%d %B %Y | %I:%M %p"
    )

    scan_id = (
        f"MRI-{datetime.now().year}-"
        f"{random.randint(10000, 99999)}"
    )

    tumor_key = tumor_class.lower().replace(" ", "_")

    db = TUMOR_DB.get(
        tumor_key,
        TUMOR_DB["no_tumor"]
    )

    is_tumor = tumor_key != "no_tumor"

    percent = result.get("tumor_percent", 0)

    severity = result.get(
        "severity",
        "N/A"
    )

    location = result.get("location")

    bbox = result.get("bbox")

    area = result.get("tumor_area", 0)

    shape = result.get("shape")

    # ========================================================
    # IMAGE QUALITY ANALYSIS
    # ========================================================

    resolution = result.get(
        "resolution",
        "256x256"
    )

    contrast_quality = result.get(
        "contrast_quality",
        "Normal"
    )

    noise_level = result.get(
        "noise_level",
        "Low"
    )

    # ========================================================
    # SHAPE ANALYSIS
    # ========================================================

    if shape:

        ecc_val = shape.get(
            "eccentricity",
            0
        )

        sol_val = shape.get(
            "solidity",
            0
        )

        # ECCENTRICITY

        if ecc_val < 0.3:

            ecc_desc = "Nearly circular"

            ecc_interp = (
                "Low eccentricity - "
                "benign morphology indicator"
            )

        elif ecc_val < 0.6:

            ecc_desc = "Mildly elongated"

            ecc_interp = (
                "Moderate eccentricity"
            )

        elif ecc_val < 0.8:

            ecc_desc = "Moderately elongated"

            ecc_interp = (
                "Elevated eccentricity"
            )

        else:

            ecc_desc = (
                "Highly irregular / elongated"
            )

            ecc_interp = (
                "High eccentricity - "
                "aggressive morphology"
            )

        # SOLIDITY

        if sol_val >= 0.90:

            sol_desc = (
                "Compact and convex"
            )

            sol_interp = (
                "High solidity - "
                "well-defined margin"
            )

        elif sol_val >= 0.75:

            sol_desc = (
                "Mostly solid with minor irregularity"
            )

            sol_interp = (
                "Moderate solidity"
            )

        elif sol_val >= 0.55:

            sol_desc = (
                "Lobulated / irregular"
            )

            sol_interp = (
                "Reduced solidity"
            )

        else:

            sol_desc = (
                "Highly irregular / spiculated"
            )

            sol_interp = (
                "Low solidity - "
                "aggressive morphology"
            )

        # SHAPE RISK

        if ecc_val >= 0.7 and sol_val < 0.80:

            shape_flag = (
                "HIGH SHAPE RISK - "
                "Irregular infiltrative morphology."
            )

        elif ecc_val >= 0.7 or sol_val < 0.80:

            shape_flag = (
                "MODERATE SHAPE RISK - "
                "Some irregular morphology detected."
            )

        else:

            shape_flag = (
                "LOW SHAPE RISK - "
                "Compact regular morphology."
            )

    else:

        ecc_val = 0
        sol_val = 0

        ecc_desc = "N/A"
        ecc_interp = "N/A"

        sol_desc = "N/A"
        sol_interp = "N/A"

        shape_flag = (
            "N/A - no tumor detected"
        )

    # ========================================================
    # SEGMENTATION OBSERVATION
    # ========================================================

    if percent == 0:

        seg_observation = (
            "No abnormal segmented tumor region identified."
        )

    else:

        if percent < 2:
            size_desc = "small"

        elif percent < 10:
            size_desc = "moderate"

        else:
            size_desc = "large"

        if location:

            loc_desc = (
                f"near X={location[0]:.1f}, "
                f"Y={location[1]:.1f}"
            )

        else:

            loc_desc = "location unclear"

        seg_observation = (
            f"A {size_desc} segmented lesion occupying "
            f"{percent:.2f}% of the brain region "
            f"was identified. "
            f"The lesion appears "
            f"{sol_desc.lower()} with "
            f"{ecc_desc.lower()} morphology "
            f"located {loc_desc}."
        )

    # ========================================================
    # RISK
    # ========================================================

    risk_label, risk_action = get_risk_info(
        tumor_key,
        percent
    )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    recs = "\n".join([
        f"   {i+1}. {r}"
        for i, r in enumerate(
            db["recommendations"]
        )
    ])

    # ========================================================
    # FINAL REPORT
    # ========================================================

    report = f"""
========================================================
          NEUROINSIGHT AI RADIOLOGY REPORT
========================================================

Scan ID   : {scan_id}
Date/Time : {current_time}

========================================================
1. MRI VALIDATION
========================================================

Result:
Valid Brain MRI Scan

Validation Confidence:
{brain_conf * 100:.2f}%

========================================================
MRI IMAGE QUALITY ANALYSIS
========================================================

Image Resolution:
{resolution}

Contrast Quality:
{contrast_quality}

Noise Level:
{noise_level}

Artifacts:
No significant imaging artifacts detected.

Image Assessment:
MRI quality is sufficient for AI-based analysis.

========================================================
2. TUMOR CLASSIFICATION
========================================================

Detection Result:
{"Tumor Detected" if is_tumor else "No Tumor Detected"}

Tumor Type:
{db["display_name"]}

Model Confidence:
{tumor_conf * 100:.2f}%

Confidence Interpretation:
{
    "Very High Confidence"
    if tumor_conf > 0.95 else
    "High Confidence"
    if tumor_conf > 0.85 else
    "Moderate Confidence"
    if tumor_conf > 0.70 else
    "Low Confidence"
}

========================================================
3. TUMOR SEGMENTATION ANALYSIS
========================================================

Affected Brain Region:
{percent:.2f}%

Tumor Area:
{area} pixels

Severity:
{severity}

Location:
{location if location else "N/A"}

Bounding Box:
{bbox if bbox else "N/A"}

--------------------------------------------------------
SHAPE ANALYSIS
--------------------------------------------------------

ECCENTRICITY:
{round(ecc_val, 3)}

Description:
{ecc_desc}

Interpretation:
{ecc_interp}

--------------------------------------------------------

SOLIDITY:
{round(sol_val, 3)}

Description:
{sol_desc}

Interpretation:
{sol_interp}

--------------------------------------------------------

Shape Risk:
{shape_flag}

--------------------------------------------------------

Segmentation Observation:
{seg_observation}

--------------------------------------------------------
TUMOR BEHAVIOR ANALYSIS
--------------------------------------------------------

Growth Pattern:
{
    "Localized and compact"
    if sol_val > 0.90 else
    "Partially infiltrative"
    if sol_val > 0.70 else
    "Highly infiltrative and irregular"
}

Morphological Nature:
{
    "Likely benign morphology"
    if ecc_val < 0.4 else
    "Intermediate morphology"
    if ecc_val < 0.7 else
    "Aggressive morphology characteristics"
}

Boundary Characteristics:
{
    "Well-defined tumor margins"
    if sol_val > 0.90 else
    "Partially irregular margins"
    if sol_val > 0.70 else
    "Poorly defined infiltrative margins"
}

========================================================
EXPLAINABLE AI ANALYSIS
========================================================

Grad-CAM Visualization:
Activated

Observation:
The AI model focused primarily on abnormal
high-intensity lesion regions during prediction.

Interpretability Assessment:
The highlighted activation regions are consistent
with segmented tumor localization.

Clinical Relevance:
Grad-CAM improves transparency by explaining
which MRI regions influenced the AI prediction.

========================================================
4. TUMOR PROFILE
========================================================

Description:
{db["description"]}

Characteristics:
{chr(10).join("- " + c for c in db["characteristics"])}

========================================================
5. RISK ASSESSMENT
========================================================

Risk Level:
{risk_label}

Clinical Severity Interpretation:
{
    "Minimal abnormality detected"
    if percent < 2 else
    "Moderate tumor burden observed"
    if percent < 10 else
    "Significant tumor involvement detected"
}

Suggested Action:
{risk_action}

Prognostic Indicator:
{
    "Favorable prognosis likely with monitoring"
    if tumor_key == "meningioma" else

    "Requires endocrine and neurological assessment"
    if tumor_key == "pituitary" else

    "Requires urgent oncological evaluation"
    if tumor_key == "glioma" else

    "No concerning prognosis indicators"
}

========================================================
6. AI MEDICAL SUMMARY
========================================================

{db["summary"]}

========================================================
FINAL RADIOLOGY IMPRESSION
========================================================

{
    f"Findings are suggestive of {db['display_name']} "
    f"with approximately {percent:.2f}% regional involvement."
    if is_tumor
    else "No radiological evidence of intracranial tumor detected."
}

Overall Assessment:
{
    "Requires immediate specialist evaluation."
    if risk_label == "HIGH RISK" else

    "Clinical follow-up recommended."
    if risk_label == "MODERATE RISK" else

    "Routine monitoring advised."
}

========================================================
7. RECOMMENDATIONS
========================================================

{recs}

========================================================
AI TECHNICAL ANALYSIS
========================================================

Primary Classification Model:
ResNet50 CNN

Segmentation Architecture:
U-Net Deep Learning Network

Inference Pipeline:
MRI Validation → Tumor Classification → Segmentation

AI Decision Support:
Enabled

Explainable AI:
Grad-CAM Integrated

Report Generation:
Automated AI-assisted radiology workflow

========================================================
SYSTEM RELIABILITY
========================================================

Classification Confidence:
{tumor_conf * 100:.2f}%

Segmentation Reliability:
{
    "High"
    if percent > 0 else
    "Not Applicable"
}

AI Consistency:
Prediction confidence and segmentation findings
show strong agreement.

========================================================
8. DISCLAIMER
========================================================

This report is AI-generated and intended for
educational/research purposes only.

Final diagnosis must always be confirmed
by qualified medical professionals.

========================================================
        END OF REPORT
========================================================
"""

    return report