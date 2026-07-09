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

# Helper untuk encode local image ke base64 agar bisa dirender custom di HTML
def get_base64_image(path):
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded_string}"
    return None

# Convert header image to base64
header_b64 = get_base64_image("static/images/header.png")

# ==========================================
# ADVANCED CUSTOM CSS INJECTION
# ==========================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
        background-color: #fafafa !important;
    }
    
    /* Custom Card container */
    .custom-card {
        background-color: #ffffff;
        padding: 2.5em;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        border: 1px solid #eaeaea;
        margin-bottom: 2em;
    }
    
    /* Styling tab Streamlit agar menyerupai Nav Bar index.html */
    button[data-baseweb="tab"] {
        font-size: 1.15rem !important;
        color: #666666 !important;
        font-weight: 500 !important;
        border-bottom: 2px solid transparent !important;
        background-color: transparent !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #ffafcc !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffafcc !important;
        border-bottom: 2px solid #ffafcc !important;
        font-weight: 600 !important;
    }
    
    /* Styling tombol agar pink pastel */
    div.stButton > button {
        background-color: #ffc8dd !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 500 !important;
        transition: background 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: block;
        margin: 0 auto;
    }
    div.stButton > button:hover {
        background-color: #ffafcc !important;
        color: #000000 !important;
    }
    
    /* Input border customization */
    input {
        border-radius: 8px !important;
        border: 1px solid #ccc !important;
        padding: 10px !important;
    }
    
    /* Title alignment */
    .custom-title-section {
        text-align: center;
        margin-bottom: 2.5em;
        margin-top: 1em;
    }
    
    /* Styling label prediksi */
    .hasil-prediksi {
        font-size: 22px;
        font-weight: 600;
        color: #555555;
        text-align: center;
        margin-top: 15px;
        background: #fdf2f4;
        padding: 12px;
        border-radius: 8px;
        border-left: 5px solid #ffafcc;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MAIN HEADER
# ==========================================
st.markdown("""
<div class="custom-title-section">
    <h1 style="color: #ffafcc; font-size: 3.2rem; font-weight: 800; margin-bottom: 0.1em; letter-spacing: -1px;">
        SIBI Sign Language Detection System
    </h1>
    <p style="color: #888; font-size: 1.1rem; font-weight: 400; max-width: 700px; margin: 0 auto;">
        Developed by Azahra Alayda Faris
    </p>
</div>
""", unsafe_allow_html=True)

# Navigasi tab utama
tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Upload Gambar", "Realtime Detection", "Text to Gesture"])

# ==========================================
# TAB 1: INTRODUCTION
# ==========================================
with tab1:
    st.markdown(f"""
    <div class="custom-card">
        <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 30px; padding: 10px 0;">
            <div style="flex: 2; min-width: 300px;">
                <h2 style="color: #333; margin-top: 0; font-weight: 700; font-size: 2rem;">SIBI Sign Language Detection System</h2>
                <p style="font-size:1.15rem; line-height:1.9; color:#555; margin-bottom: 1.5em;">
                A web-based application designed to detect sign language (SIBI) in real-time using the YOLOv8 model. 
                This system is also equipped with a feature that translates letters into visual gestures, helping users 
                to understand and learn sign language in a more interactive and accessible way.
                </p>
            </div>
            <div style="flex: 1; min-width: 250px; display: flex; justify-content: center;">
                <img src="{header_b64}" style="width: 300px; max-width: 100%; border-radius: 8px;" />
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# TAB 2: UPLOAD GAMBAR
# ==========================================
with tab2:
    st.markdown("""
    <div class="custom-card" style="text-align: center;">
        <h2 style="color: #333; margin-top: 0; font-weight: 700; font-size: 2rem;">Upload Gambar</h2>
        <p style="color: #888; margin-bottom: 0px;">*Please upload an image in JPG or PNG format for optimal detection results.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Drag and drop upload
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # YOLO inference
        results = model(img_array)
        annotated_img = results[0].plot()
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        pred = results[0].names[int(results[0].boxes.cls[0])] if results[0].boxes else "No detection"
        
        # Grid tampil
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2:
            st.image(annotated_img_rgb, use_column_width=True)
            st.markdown(f'<div class="hasil-prediksi">{pred}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: REALTIME DETECTION
# ==========================================
with tab3:
    st.markdown("""
    <div class="custom-card" style="text-align: center;">
        <h2 style="color: #333; margin-top: 0; font-weight: 700; font-size: 2rem;">Realtime Detection</h2>
        <p style="color: #888; margin-bottom: 0px;">Use your webcam to take a photo of SIBI sign language gesture.</p>
    </div>
    """, unsafe_allow_html=True)
    
    img_file_buffer = st.camera_input("Ambil foto menggunakan webcam", label_visibility="collapsed")
    
    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        img_array = np.array(img)
        
        # YOLO inference
        results = model(img_array)
        annotated_img = results[0].plot()
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        pred = results[0].names[int(results[0].boxes.cls[0])] if results[0].boxes else "No detection"
        
        # Grid tampil
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        col_cam1, col_cam2, col_cam3 = st.columns([1, 2, 1])
        with col_cam2:
            st.image(annotated_img_rgb, use_column_width=True)
            st.markdown(f'<div class="hasil-prediksi">{pred}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 4: TEXT TO GESTURE
# ==========================================
with tab4:
    st.markdown("""
    <div class="custom-card" style="text-align: center;">
        <h2 style="color: #333; margin-top: 0; font-weight: 700; font-size: 2rem;">Text to Gesture</h2>
        <p style="color: #888; margin-bottom: 20px;">Enter letters to translate them into SIBI sign language gestures.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input box
    text_input = st.text_input("", placeholder="Enter letters...", label_visibility="collapsed").upper()
    
    if text_input:
        html_images = ""
        for char in text_input:
            if char.isalpha():
                img_path = f"static/images/{char}.jpg"
                b64_data = get_base64_image(img_path)
                if b64_data:
                    # Rendering custom image card dengan border radius & bayangan halus
                    html_images += f"""
                    <img src="{b64_data}" style="
                        margin: 8px; 
                        border-radius: 8px; 
                        box-shadow: 0 4px 10px rgba(0,0,0,0.15); 
                        width: 170px;
                        border: 1px solid #eee;
                    " />
                    """
        
        if html_images:
            st.markdown(f"""
            <div class="custom-card">
                <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; padding: 10px;">
                    {html_images}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No visual gesture found for characters input.")
            
# Footer copyright
st.markdown("""
<div style="text-align: center; margin-top: 4em; color: #888; font-size: 0.9em; border-top: 1px solid #eee; padding-top: 2em; padding-bottom: 2em;">
    <p>
        © 2026 SIBI Sign Language Detection System <br>
        Developed by Azahra Alayda Faris
    </p>
</div>
""", unsafe_allow_html=True)