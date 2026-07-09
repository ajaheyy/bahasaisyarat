import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import base64

# Page configuration
st.set_page_config(
    page_title="Sign Language Detection System",
    page_icon="🤟",
    layout="wide"
)

# Load model (cached)
@st.cache_resource
def load_model():
    return YOLO("model/my_model.pt")

model = load_model()

# Helper to encode local image to base64 for custom styled HTML rendering
def get_base64_image(path):
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded_string}"
    return None

# ==========================================
# CUSTOM CSS INJECTION (MATCHING THE ORIGINAL THEME AESTHETIC)
# ==========================================
st.markdown("""
<style>
    /* Styling page background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Styling Streamlit buttons with original pink theme colors */
    div.stButton > button {
        background-color: #ffc8dd !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 500 !important;
        transition: background 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div.stButton > button:hover {
        background-color: #ffafcc !important;
        color: #000000 !important;
    }
    
    /* Input border customization */
    input {
        border-radius: 8px !important;
        border: 1px solid #ccc !important;
    }
    
    /* Center align main header block */
    .custom-title-section {
        text-align: center;
        margin-bottom: 2em;
    }
    
    /* Output predicted label card styling */
    .hasil-prediksi {
        font-size: 22px;
        font-weight: 600;
        color: #555555;
        text-align: center;
        margin-top: 15px;
        background: #fdf2f4;
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #ffafcc;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER SECTION
# ==========================================
st.markdown("""
<div class="custom-title-section">
    <h1 style="color: #ffafcc; font-size: 2.8rem; font-weight: 800; margin-bottom: 0.1em;">
        SIBI Sign Language Detection System
    </h1>
    <p style="color: #666; font-size: 1.1rem; max-width: 700px; margin: 0 auto;">
        Developed by Azahra Alayda Faris
    </p>
</div>
""", unsafe_allow_html=True)

# Primary navigation tabs matching the original HTML index navigation
tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Upload Gambar", "Realtime Detection", "Text to Gesture"])

# ==========================================
# TAB 1: INTRODUCTION
# ==========================================
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<h2 style='color:#333;'>Introduction</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:1.1rem; line-height:1.8; color:#555; max-width:650px;">
        A web-based application designed to detect sign language (SIBI) in real-time using the YOLOv8 model. 
        This system is also equipped with a feature that translates letters into visual gestures, helping users 
        to understand and learn sign language in a more interactive and accessible way.
        </p>
        """, unsafe_allow_html=True)
    with col2:
        header_img_path = "static/images/header.png"
        if os.path.exists(header_img_path):
            st.image(header_img_path, width=280)

# ==========================================
# TAB 2: UPLOAD GAMBAR
# ==========================================
with tab2:
    st.markdown("<h2 style='text-align:center; color:#333;'>Upload Gambar</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#888;'>*Please upload an image in JPG or PNG format for optimal detection results.</p>", unsafe_allow_html=True)
    
    # Hide the raw Streamlit label parameter
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Run inference
        results = model(img_array)
        annotated_img = results[0].plot()
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        pred = results[0].names[int(results[0].boxes.cls[0])] if results[0].boxes else "No detection"
        
        # Centered output preview
        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2:
            st.image(annotated_img_rgb, use_column_width=True)
            st.markdown(f'<div class="hasil-prediksi">Prediction: {pred}</div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: REALTIME DETECTION
# ==========================================
with tab3:
    st.markdown("<h2 style='text-align:center; color:#333;'>Realtime Detection</h2>", unsafe_allow_html=True)
    
    img_file_buffer = st.camera_input("Ambil foto menggunakan webcam", label_visibility="collapsed")
    
    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        img_array = np.array(img)
        
        # Run inference
        results = model(img_array)
        annotated_img = results[0].plot()
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        pred = results[0].names[int(results[0].boxes.cls[0])] if results[0].boxes else "No detection"
        
        # Centered output preview
        col_cam1, col_cam2, col_cam3 = st.columns([1, 2, 1])
        with col_cam2:
            st.image(annotated_img_rgb, use_column_width=True)
            st.markdown(f'<div class="hasil-prediksi">Prediction: {pred}</div>', unsafe_allow_html=True)

# ==========================================
# TAB 4: TEXT TO GESTURE
# ==========================================
with tab4:
    st.markdown("<h2 style='text-align:center; color:#333;'>Text to Gesture</h2>", unsafe_allow_html=True)
    
    # Text input styled to mimic input field from original templates
    text_input = st.text_input("", placeholder="Enter letters...", label_visibility="collapsed").upper()
    
    if text_input:
        html_images = ""
        for char in text_input:
            if char.isalpha():
                img_path = f"static/images/{char}.jpg"
                b64_data = get_base64_image(img_path)
                if b64_data:
                    # Renders images precisely with the same shadow/radius style
                    html_images += f"""
                    <img src="{b64_data}" style="
                        margin: 5px; 
                        border-radius: 8px; 
                        box-shadow: 0 2px 6px rgba(0,0,0,0.2); 
                        width: 160px;
                    " />
                    """
        
        if html_images:
            st.markdown(f"""
            <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; padding: 20px;">
                {html_images}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No visual gesture found for characters input.")
            
# Footer copyright matching index.html
st.markdown("""
<div style="text-align: center; margin-top: 3em; color: #888; font-size: 0.9em; border-top: 1px solid #eee; padding-top: 1.5em;">
    <p>
        © 2026 SIBI Sign Language Detection System <br>
        Developed by Azahra Alayda Faris
    </p>
</div>
""", unsafe_allow_html=True)