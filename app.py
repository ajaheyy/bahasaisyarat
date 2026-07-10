import os
import cv2
import numpy as np
import base64
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Pastikan folder static ada untuk menyimpan file upload
os.makedirs("static", exist_ok=True)

# Load ONNX model menggunakan ONNX Runtime (jauh lebih cepat dari OpenCV DNN)
# Optimize session options untuk speed maksimal
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = os.cpu_count() or 4
opts.inter_op_num_threads = 2
opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL

session = ort.InferenceSession(
    "model/my_model.onnx",
    providers=["CPUExecutionProvider"],
    sess_options=opts
)
input_name = session.get_inputs()[0].name

# Daftar nama kelas SIBI (A-Z)
CLASSES = [chr(i) for i in range(65, 91)]  # ['A', 'B', ..., 'Z']

def run_inference(img, draw_on_image=True):
    """Run YOLOv8 inference on an image using ONNX Runtime."""
    h, w = img.shape[:2]
    
    # Preprocessing: resize, normalize, BGR->RGB, HWC->CHW, add batch dim
    resized = cv2.resize(img, (320, 320))
    blob = resized.astype(np.float32) / 255.0
    blob = blob[:, :, ::-1]  # BGR -> RGB
    blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
    blob = np.expand_dims(blob, axis=0)  # (1, 3, 320, 320)
    blob = np.ascontiguousarray(blob)
    
    # Run ONNX Runtime inference
    outputs = session.run(None, {input_name: blob})
    
    # Vectorized post-processing
    predictions = outputs[0][0].T  # shape: (2100, 30)
    
    scores_matrix = predictions[:, 4:]
    class_ids = np.argmax(scores_matrix, axis=1)
    max_scores = scores_matrix[np.arange(len(class_ids)), class_ids]
    
    # Filter by confidence threshold
    mask = max_scores > 0.4
    if not np.any(mask):
        return img, "No detection", None
    
    # Get best detection
    filtered_indices = np.where(mask)[0]
    best_idx = filtered_indices[np.argmax(max_scores[mask])]
    
    best_class_id = int(class_ids[best_idx])
    best_score = float(max_scores[best_idx])
    best_box = predictions[best_idx, 0:4]
    
    pred_label = CLASSES[best_class_id]
    
    # YOLOv8 format: [x_center, y_center, width, height] relative to 320x320
    x_center, y_center, box_w, box_h = best_box
    x_factor = w / 320.0
    y_factor = h / 320.0
    
    left = int((x_center - box_w / 2) * x_factor)
    top = int((y_center - box_h / 2) * y_factor)
    bw = int(box_w * x_factor)
    bh = int(box_h * y_factor)
    
    box_coords = [left, top, bw, bh]
    
    if draw_on_image:
        cv2.rectangle(img, (left, top), (left + bw, top + bh), (180, 200, 255), 3)
        cv2.putText(img, f"{pred_label} ({best_score:.2f})", (left, top - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 200, 255), 2)
    
    return img, pred_label, box_coords

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

    img = cv2.imread(filepath)
    annotated_img, pred, _ = run_inference(img)
    cv2.imwrite("static/result.jpg", annotated_img)

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

    _, pred, box = run_inference(img, draw_on_image=False)

    return jsonify({
        "prediction": pred,
        "box": box
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, host="0.0.0.0", port=port)