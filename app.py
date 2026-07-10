import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO

# Optimasi PyTorch CPU Threading untuk mencegah CPU contention/overhead di web server
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

app = Flask(__name__)

# Pastikan folder static ada untuk menyimpan file upload
os.makedirs("static", exist_ok=True)

# Load YOLOv8 model
model = YOLO("model/my_model.pt")

# Daftar nama kelas SIBI (A-Z)
CLASSES = [chr(i) for i in range(65, 91)]  # ['A', 'B', ..., 'Z']

@app.route('/')
def home():
    return render_template("index.html")

# ======================
# UPLOAD GAMBAR
# ======================
@app.route('/predict-image', methods=['POST'])
def predict_image():
    file = request.files['image']
    filepath = "static/upload.jpg"
    file.save(filepath)

    # Tetap gunakan 320 untuk upload gambar agar lebih presisi
    results = model(filepath, imgsz=320, verbose=False)
    results[0].save(filename="static/result.jpg")

    pred = "No detection"
    if results[0].boxes and len(results[0].boxes) > 0:
        pred = results[0].names[int(results[0].boxes.cls[0])]

    return jsonify({
        "prediction": pred,
        "image": "/static/result.jpg"
    })

# ======================
# WEBCAM FRAME DETECTION
# ======================
@app.route('/predict-frame', methods=['POST'])
def predict_frame():
    data = request.json['image']

    # Decode base64
    encoded = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Gunakan imgsz=224 (lebih kecil -> jauh lebih cepat di CPU, ~2x-3x speedup)
    results = model.predict(img, imgsz=224, verbose=False)

    pred = "No detection"
    box = None

    if results[0].boxes and len(results[0].boxes) > 0:
        # Ambil deteksi confidence tertinggi
        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))

        pred = results[0].names[int(boxes.cls[best_idx])]

        # Koordinat [x1, y1, x2, y2] -> [left, top, width, height]
        xyxy = boxes.xyxy.cpu().numpy()[best_idx]
        left = int(xyxy[0])
        top = int(xyxy[1])
        bw = int(xyxy[2] - xyxy[0])
        bh = int(xyxy[3] - xyxy[1])
        box = [left, top, bw, bh]

    return jsonify({
        "prediction": pred,
        "box": box
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, host="0.0.0.0", port=port)