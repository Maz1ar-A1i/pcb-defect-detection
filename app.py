from flask import Flask, request, render_template, jsonify
import torch
from ultralytics import YOLO
import os
import cv2
import uuid

# Initialize Flask app
app = Flask(__name__)

# Reduce CPU thread usage to avoid OOM on small instances
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass

# Load your trained YOLO model once at startup (CPU)
model = YOLO("best.pt")  # change path if your weights are elsewhere
model.to("cpu")

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
    try:
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

        # Downscale very large images to reduce memory/CPU usage
        try:
            import PIL.Image as Image
            with Image.open(file_path) as im:
                im = im.convert("RGB")
                max_side = 1280
                w, h = im.size
                scale = min(1.0, max_side / max(w, h))
                if scale < 1.0:
                    new_size = (int(w * scale), int(h * scale))
                    im = im.resize(new_size, Image.LANCZOS)
                    im.save(file_path, format="JPEG", quality=90)
        except Exception:
            pass

        # Run YOLO detection (no auto-save). We'll plot and save once to reduce IO/memory.
        run_name = os.path.splitext(filename)[0]
        results = model.predict(
            file_path,
            device="cpu",
            imgsz=640,
            conf=0.25,
            save=False,
            verbose=False,
        )

        # Prepare output directory and save annotated image
        save_dir = os.path.join(RESULTS_FOLDER, run_name)
        os.makedirs(save_dir, exist_ok=True)
        annotated_filename = os.path.basename(file_path)
        result_img_path = os.path.join(save_dir, annotated_filename)
        try:
            import numpy as np
            import cv2 as _cv2
            annotated = results[0].plot()  # numpy array, BGR
            _cv2.imwrite(result_img_path, annotated)
        except Exception as _plot_exc:
            # Fallback: if plotting fails, just copy the original
            import shutil
            shutil.copyfile(file_path, result_img_path)

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
    except Exception as exc:
        # Log full exception to server logs and return JSON error
        print("/predict error:", exc)
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
