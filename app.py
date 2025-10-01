from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
import os
import cv2
import uuid

# Initialize Flask app
app = Flask(__name__)

# Load your trained YOLO model once at startup
model = YOLO("best.pt")  # change path if your weights are elsewhere

# Ensure uploads folder exists
UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")  # frontend template (we’ll add later)

@app.route("/predict", methods=["POST"])
def predict():
    # Accept either 'file' (upload) or 'image' (webcam blob)
    uploaded = None
    if "file" in request.files:
        uploaded = request.files["file"]
    elif "image" in request.files:
        uploaded = request.files["image"]

    if uploaded is None:
        return jsonify({"error": "No image provided. Use form-data with key 'file' or 'image'."}), 400

    # Some browsers may send empty filename for blobs; normalize
    original_filename = uploaded.filename or "capture.jpg"
    ext = os.path.splitext(original_filename)[1].lower() or ".jpg"
    filename = f"{uuid.uuid4()}{ext}"

    # Save uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded.save(file_path)

    # Run YOLO detection and save results to a dedicated subfolder per request
    run_name = os.path.splitext(filename)[0]
    results = model.predict(file_path, save=True, project=RESULTS_FOLDER, name=run_name)

    # YOLO saves the annotated image under RESULTS_FOLDER/run_name/<original_filename>
    result_dir = os.path.join(RESULTS_FOLDER, run_name)
    annotated_filename = os.path.basename(file_path)
    result_img_path = os.path.join(result_dir, annotated_filename)

    # Normalize paths to URL format for frontend (Windows-safe)
    def to_url_path(path):
        path = path.replace("\\", "/")
        if not path.startswith("/"):
            path = "/" + path
        return path

    response = {
        "input_image": to_url_path(file_path),
        "result_image": to_url_path(result_img_path),
        "detections": results[0].boxes.xyxy.tolist() if hasattr(results[0].boxes, "xyxy") else [],
        "labels": results[0].boxes.cls.tolist() if hasattr(results[0].boxes, "cls") else [],
        "scores": results[0].boxes.conf.tolist() if hasattr(results[0].boxes, "conf") else [],
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
