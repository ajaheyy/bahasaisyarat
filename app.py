import os
import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr
import spaces

# Load model
model = YOLO("model/my_model.pt")

@spaces.GPU
def predict_image(img):
    if img is None:
        return None, "No image"
    
    # YOLO inference (runs on GPU)
    results = model(img)
    
    # Plot detection results (BGR format)
    annotated = results[0].plot()
    
    # Convert BGR to RGB so it displays correctly in Gradio
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    pred = results[0].names[int(results[0].boxes.cls[0])] if results[0].boxes else "No detection"
    return annotated_rgb, pred

def text_to_gesture(text):
    if not text:
        return []
    images = []
    for char in text.upper():
        if char.isalpha():
            img_path = f"static/images/{char}.jpg"
            if os.path.exists(img_path):
                images.append(img_path)
    return images

# Custom CSS for beautiful styling matching the original aesthetic
custom_css = """
body {
    background-color: #fafafa;
}
.title-header {
    text-align: center;
    margin-bottom: 25px;
}
.title-header h1 {
    color: #ffafcc !important;
    font-weight: 800;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""
    <div class="title-header">
        <h1>SIBI Sign Language Detection System</h1>
        <p>A web-based system designed to detect SIBI sign language using YOLOv8 on ZeroGPU, equipped with Text-to-Gesture translation.</p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.TabItem("Introduction & Text to Gesture"):
            gr.Markdown("### Text to Gesture Translation\nEnter text to translate it into SIBI sign language gestures.")
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(label="Enter letters (e.g. HELLO)", placeholder="Type here...")
                    btn_generate = gr.Button("Generate Gestures", variant="primary")
                with gr.Column():
                    gallery_output = gr.Gallery(label="Sign Language Gestures", columns=4, height="auto")
            
            btn_generate.click(fn=text_to_gesture, inputs=text_input, outputs=gallery_output)
            
        with gr.TabItem("Image Upload Detection"):
            gr.Markdown("### Upload Image\nUpload an image containing SIBI sign language gesture for detection.")
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="Upload Image", type="numpy")
                    btn_predict = gr.Button("Predict Image", variant="primary")
                with gr.Column():
                    image_output = gr.Image(label="Detection Result")
                    label_output = gr.Textbox(label="Prediction Label")
            
            btn_predict.click(fn=predict_image, inputs=image_input, outputs=[image_output, label_output])
            
        with gr.TabItem("Realtime Webcam Detection"):
            gr.Markdown("### Realtime Webcam\nAllow camera access to detect SIBI sign language in real-time using GPU acceleration.")
            with gr.Row():
                with gr.Column():
                    webcam_input = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam Stream")
                with gr.Column():
                    webcam_output = gr.Image(label="Live Prediction")
                    webcam_label = gr.Textbox(label="Live Label")
            
            webcam_input.stream(fn=predict_image, inputs=webcam_input, outputs=[webcam_output, webcam_label], queue=False)

if __name__ == "__main__":
    demo.queue().launch()