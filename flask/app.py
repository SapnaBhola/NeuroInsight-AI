from flask import Flask, render_template, request, redirect, session, send_file
from flask_mail import Mail, Message
import os
import uuid
import json
import base64
import re

from PIL import Image
import cv2

import brain
import tumor
import segment

from report_generator import generate_ai_report
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

print("🔥 APP.PY IS RUNNING")

app = Flask(__name__)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True

app.config['MAIL_USERNAME'] = 'sonubhola2345@gmail.com'
app.config['MAIL_PASSWORD'] = 'aodw uhke uysu zbos'

mail = Mail(app)
# =========================================================
# SECRET KEY
# =========================================================

app.secret_key = "neuroinsight_secret_key"

# =========================================================
# PATH SETUP
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HISTORY_FILE = os.path.join(
    BASE_DIR,
    "history.json"
)

USERS_FILE = os.path.join(
    BASE_DIR,
    "users.json"
)

CONTACT_FILE = os.path.join(
    BASE_DIR,
    "contacts.json"
)

# ✅ FIX: REPORT FOLDER (WAS MISSING)
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_FOLDER, exist_ok=True)

# =========================================================
# IMAGE -> BASE64
# =========================================================

def image_to_base64(filepath):

    with open(filepath, "rb") as image_file:

        encoded = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"

# =========================================================
# SAVE HISTORY
# =========================================================

def save_history(result):

    try:

        if os.path.exists(HISTORY_FILE):

            with open(HISTORY_FILE, "r") as f:

                data = json.load(f)

        else:

            data = []

        result["created_at"] = datetime.now().strftime(
            "%d-%m-%Y %I:%M %p"
        )

        clean_result = json.loads(
            json.dumps(result, default=str)
        )

        data.append(clean_result)

        with open(HISTORY_FILE, "w") as f:

            json.dump(data, f, indent=4)

    except Exception as e:

        print("❌ History save error:", e)

def save_contact(contact_data):

    if os.path.exists(CONTACT_FILE):

        with open(CONTACT_FILE, "r") as f:
            data = json.load(f)

    else:

        data = []

    data.append(contact_data)

    with open(CONTACT_FILE, "w") as f:
        json.dump(data, f, indent=4)
# =========================================================
# HOME PAGE
# =========================================================

@app.route("/home")
def home():

    return render_template("home.html")

# =========================================================
# DETECTION PAGE
# =========================================================

@app.route("/detect", methods=["GET", "POST"])
def detect():

    if "user" not in session:
        return redirect("/login")

    result = None

    if request.method == "POST":

        file = request.files.get("file")

        if not file or file.filename == "":

            return render_template(
                "index.html",
                result=None
            )

        try:

            uid = str(uuid.uuid4())

            filepath = os.path.join(
                UPLOAD_FOLDER,
                f"{uid}.jpg"
            )

            file.save(filepath)

            image = Image.open(
                filepath
            ).convert("RGB")

            result = {

                "image": f"/static/uploads/{uid}.jpg",

                "image_base64": image_to_base64(
                    filepath
                )

            }

            # =========================================================
            # BRAIN VALIDATION
            # =========================================================

            brain_label, brain_conf = (
                brain.predict_brain(image)
            )

            result["brain"] = {

                "label": brain_label,

                "confidence": round(
                    brain_conf,
                    4
                )

            }

            if "NOT BRAIN" in brain_label:

                result["report"] = f"""
RADIOLOGY REPORT (AI GENERATED)

Findings:
The uploaded image is not identified as a valid brain MRI scan.

Impression:
Analysis cannot be performed on non-brain images.

Confidence:
{brain_conf:.2f}

Recommendation:
Please upload a proper brain MRI image.

Note:
AI-generated report — not a clinical diagnosis.
"""

                save_history(result)

                return render_template(
                    "index.html",
                    result=result
                )

            # =========================================================
            # TUMOR CLASSIFICATION
            # =========================================================

            tumor_class, tumor_conf, pred, cam_img = (
                tumor.predict_with_gradcam(image)
            )

            result["tumor"] = {

                "class": tumor_class,

                "confidence": round(
                    tumor_conf,
                    4
                )

            }

            if tumor_conf > 0.85:

                interpretation = (
                    "High likelihood of tumor presence"
                )

            elif tumor_conf > 0.60:

                interpretation = (
                    "Moderate likelihood of tumor presence"
                )

            else:

                interpretation = (
                    "Low confidence prediction"
                )

            result["interpretation"] = interpretation

            result["uncertainty"] = round(
                1 - tumor_conf,
                4
            )

            # =========================================================
            # GRADCAM SAVE
            # =========================================================

            cam_img = cam_img.astype("uint8")

            cam_filename = f"cam_{uid}.jpg"

            cam_path = os.path.join(
                UPLOAD_FOLDER,
                cam_filename
            )

            cv2.imwrite(cam_path, cam_img)

            result["cam"] = (
                f"/static/uploads/{cam_filename}"
            )

            result["cam_base64"] = (
                image_to_base64(cam_path)
            )

            # =========================================================
            # SEGMENTATION
            # =========================================================

            if tumor_class != "No Tumor":

                _, _, overlay, res, report = (
                    segment.run_segmentation(image)
                )

                overlay = overlay.astype("uint8")

                seg_filename = f"seg_{uid}.jpg"

                seg_path = os.path.join(
                    UPLOAD_FOLDER,
                    seg_filename
                )

                cv2.imwrite(seg_path, overlay)

                result["seg"] = (
                    f"/static/uploads/{seg_filename}"
                )

                result["seg_base64"] = (
                    image_to_base64(seg_path)
                )

                result["analysis"] = res

                result["report"] = report

            else:

                result["analysis"] = {

                    "tumor_percent": 0,
                    "tumor_area": 0,
                    "bbox": "N/A",
                    "shape": "N/A",
                    "severity": "None",
                    "location": None

                }

                result["report"] = f"""
RADIOLOGY REPORT (AI GENERATED)

Findings:
No tumor detected.

Confidence:
{tumor_conf:.2f}
"""

            save_history(result)

        except Exception as e:
            result = {"error": str(e)}

    return render_template("index.html", result=result)

# =========================================================
# ROOT ROUTE
# =========================================================

@app.route("/")
def landing():

    return render_template("home.html")

# =========================================================
# SIGNUP PAGE
# =========================================================

@app.route("/signup", methods=["GET", "POST"])
def signup():

    error = None
    success = None

    if request.method == "POST":

        doctor_name = request.form.get("doctor_name")
        email = request.form.get("email")
        phone = request.form.get("phone")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if password != confirm_password:
            error = "Passwords do not match"
            return render_template("signup.html", error=error)

        password_pattern = r'^(?=.*[A-Z])(?=.*[0-9])(?=.*[\W_]).{8,}$'

        if not re.match(password_pattern, password):
            error = "Password must contain minimum 8 characters, 1 uppercase letter, 1 number and 1 special character"
            return render_template("signup.html", error=error)

        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r") as f:
                users = json.load(f)
        else:
            users = []

        for user in users:
            if user["email"] == email:
                error = "Email already registered"
                return render_template("signup.html", error=error)

        new_user = {
            "doctor_name": doctor_name,
            "email": email,
            "phone": phone,
            "password": password
        }

        users.append(new_user)

        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=4)

        success = "Account created successfully"
        return render_template("signup.html", success=success)

    return render_template("signup.html", error=error, success=success)

# =========================================================
# LOGIN PAGE
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    error = None

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r") as f:
                users = json.load(f)
        else:
            users = []

        found_user = None

        for user in users:
            if user["email"] == email:
                found_user = user
                break

        if found_user is None:
            error = "Email is wrong"
            return render_template("login.html", error=error)

        if found_user["password"] != password:
            error = "Password is wrong"
            return render_template("login.html", error=error)

        session["user"] = {
            "doctor_name": found_user["doctor_name"],
            "email": found_user["email"]
        }

        return redirect("/detect")

    return render_template("login.html", error=error)

# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect("/")

# =========================================================
# HISTORY PAGE
# =========================================================

@app.route("/history")
def history():

    if "user" not in session:
        return redirect("/login")

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    # 🔥 FIX: latest first
    data = list(reversed(data))

    return render_template("history.html", data=data)

# =========================================================
# ABOUT
# =========================================================

@app.route("/about")
def about():
    return render_template("about.html")

# =========================
# ================================
# CONTACT
# =========================================================

@app.route("/contact", methods=["GET", "POST"])
def contact():

    success = None
    error = None

    if request.method == "POST":

        name = request.form.get("name")
        email = request.form.get("email")
        subject = request.form.get("subject")
        message = request.form.get("message")

        # ==========================
        # SAVE TO contacts.json
        # ==========================

        contact_data = {
            "name": name,
            "email": email,
            "subject": subject,
            "message": message,
            "date": datetime.now().strftime(
                "%d-%m-%Y %I:%M %p"
            )
        }

        save_contact(contact_data)

        # ==========================
        # SEND EMAIL
        # ==========================

        try:

            msg = Message(
                subject=f"NeuroInsight Contact Form - {subject}",
                sender=app.config["MAIL_USERNAME"],
                recipients=["sonubhola234@gmail.com"]
            )

            msg.body = f"""
Name: {name}

Email: {email}

Subject: {subject}

Message:
{message}
"""

            mail.send(msg)

            success = """
Thank you for contacting NeuroInsight AI.

Your message has been successfully submitted to our team.

We appreciate your inquiry and will review your message as soon as possible. Our team will get back to you shortly via email if a response is required.

Thank you for choosing NeuroInsight AI.
"""

        except Exception as e:

            print("EMAIL ERROR:", e)

            error = str(e)

    return render_template(
        "contact.html",
        success=success,
        error=error
    )
# =========================================================
# REPORT PAGE
# =========================================================

@app.route("/report/<int:index>")
def report(index):

    if "user" not in session:
        return redirect("/login")

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
    else:
        return "No history"

    if index < 0 or index >= len(data):
        return "Invalid ID"

    result = data[index]

    return render_template("index.html", result=result)

# =========================================================
# =========================================================
# PDF GENERATOR
# =========================================================

def generate_pdf_report(report_text, output_path):

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    story = []

    for line in report_text.split("\n"):

        # Empty line
        if line.strip() == "":

            story.append(
                Spacer(1, 8)
            )

        else:

            paragraph = Paragraph(
                line.replace(" ", "&nbsp;"),
                styles["BodyText"]
            )

            story.append(paragraph)

    doc.build(story)


# =========================================================
# =========================================================
# DOWNLOAD REPORT (ONLY IF TUMOR EXISTS)
# =========================================================

@app.route("/download-report")
def download_report():

    if "user" not in session:
        return redirect("/login")

    try:

        # =====================================================
        # CHECK HISTORY FILE
        # =====================================================

        if not os.path.exists(HISTORY_FILE):
            return "No report found"

        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)

        if not data:
            return "No report available"

        # =====================================================
        # GET LATEST RESULT
        # =====================================================

        latest_result = data[-1]

        # =====================================================
        # CHECK TUMOR
        # =====================================================

        tumor_class = str(

    latest_result.get(
        "tumor",
        {}
    ).get(
        "class",
        "No Tumor"
    )

).strip().lower().replace("_", " ")
        

        # ✅ IMPORTANT CONDITION
        if tumor_class.lower() == "no tumor":

            return """
            <h2 style='color:red;text-align:center;margin-top:50px;'>
            No Tumor Detected — Report Download Disabled
            </h2>
            """

        # =====================================================
        # SAFE ANALYSIS DATA
        # =====================================================

        analysis = latest_result.get(
            "analysis",
            {}
        ) or {}

        shape = (
            analysis.get("shape", {})
            if isinstance(
                analysis.get("shape"),
                dict
            )
            else {}
        )

        safe_result = {

            "tumor_percent": analysis.get(
                "tumor_percent",
                0
            ),

            "tumor_area": analysis.get(
                "tumor_area",
                0
            ),

            "bbox": analysis.get(
                "bbox",
                "N/A"
            ),

            "severity": analysis.get(
                "severity",
                "N/A"
            ),

            "location": analysis.get(
                "location",
                None
            ),

            "shape": {

                "eccentricity": shape.get(
                    "eccentricity",
                    0
                ),

                "solidity": shape.get(
                    "solidity",
                    0
                )
            }
        }

        # =====================================================
        # GENERATE AI REPORT
        # =====================================================

        report_text = generate_ai_report(

            latest_result.get(
                "brain",
                {}
            ).get(
                "label",
                "N/A"
            ),

            latest_result.get(
                "brain",
                {}
            ).get(
                "confidence",
                0
            ),

            tumor_class,

            latest_result.get(
                "tumor",
                {}
            ).get(
                "confidence",
                0
            ),

            safe_result
        )

        # =====================================================
        # CREATE PDF FILE
        # =====================================================

        file_id = str(uuid.uuid4())

        file_path = os.path.join(
            REPORT_FOLDER,
            f"{file_id}.pdf"
        )

        generate_pdf_report(
            report_text,
            file_path
        )

        # =====================================================
        # SEND PDF
        # =====================================================

        return send_file(

            file_path,

            as_attachment=True,

            download_name="NeuroInsight_AI_Report.pdf"
        )

    except Exception as e:

        return f"Error generating report: {str(e)}"
# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    print("🔥 Flask starting...")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader = False
    )

   