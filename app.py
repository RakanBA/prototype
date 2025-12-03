import streamlit as st
import tensorflow as tf
import requests
import os
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image, ImageOps

# ==========================================
# 1. PAGE CONFIG & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="Heritage Vision AI",
    page_icon="🕌", # Mosque/Heritage icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to force the "History" vibe
st.markdown("""
    <style>
    /* Import a nice font */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap');
    
    h1, h2, h3 { font-family: 'Cinzel', serif; color: #4A3B32; }
    
    .stApp {
        background-image: linear-gradient(to bottom right, #ffffff, #fdfbf7);
    }
    
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e0dace;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA: HISTORICAL INFO (MOCK DB)
# ==========================================
# This acts as your "Database" of facts. 
# You should ensure these keys MATCH your class_names exactly.
LANDMARK_INFO = {
    "Nasseef House": {
        "desc": "Built in 1881, Nasseef House is historically significant as the residence of King Abdulaziz when he entered Jeddah in 1925.",
        "location": "Old Jeddah (Al-Balad)",
        "coords": [21.4833, 39.1833] # Lat, Long
    },
    "Al-Alawi Souq": {
        "desc": "One of the oldest markets in the region, connecting the port to the Makkah Gate. Famous for spices and incense.",
        "location": "Heart of Al-Balad",
        "coords": [21.4850, 39.1870]
    },
    # Add a default fallback
    "Unknown": {
        "desc": "Historical data for this landmark is currently being curated.",
        "location": "Jeddah, Saudi Arabia",
        "coords": [21.5433, 39.1728]
    }
}

# ==========================================
# 3. LOAD ASSETS (Cached)
# ==========================================
MODEL_URL = "https://huggingface.co/RakanBA/heritage-vision-v1/resolve/main/master_classifier.tflite?download=true"
LABELS_URL = "https://huggingface.co/RakanBA/heritage-vision-v1/resolve/main/labels.txt?download=true"
LOCAL_MODEL = "master_classifier.tflite"
LOCAL_LABELS = "labels.txt"

@st.cache_resource
def load_assets():
    if not os.path.exists(LOCAL_LABELS):
        with open(LOCAL_LABELS, 'wb') as f: f.write(requests.get(LABELS_URL).content)
    if not os.path.exists(LOCAL_MODEL):
        with open(LOCAL_MODEL, 'wb') as f: f.write(requests.get(MODEL_URL).content)

    with open(LOCAL_LABELS, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    interpreter = tf.lite.Interpreter(model_path=LOCAL_MODEL)
    interpreter.allocate_tensors()
    return interpreter, classes

interpreter, class_names = load_assets()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==========================================
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4342/4342728.png", width=80)
    st.title("Heritage Vision")
    st.caption("Preserving Jeddah's History with AI")
    st.markdown("---")
    st.write("### 📂 Project Details")
    st.info("This prototype demonstrates the potential of computer vision in digital tourism and heritage preservation.")

# ==========================================
# 5. MAIN INTERFACE
# ==========================================
col_header, col_logo = st.columns([4, 1])
with col_header:
    st.title("Landmark Recognition")
    st.write("Identify historical architecture in Al-Balad.")

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # --- PROCESSING ---
    image = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner("Consulting the digital archive..."):
        # Preprocess
        img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        input_data = np.array(img_resized, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Results
        top_index = np.argmax(output_data)
        top_label = class_names[top_index]
        top_conf = output_data[top_index]

    # --- UI LAYOUT ---
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.image(image, use_container_width=True, caption="Your Upload")
        
        # UX: Success Animation if high confidence
        if top_conf > 0.85:
            st.balloons()

    with col2:
        # Result Header
        st.markdown(f"""
        <div class="result-card">
            <p style="color:#888; font-size: 14px; margin-bottom:0;">Identified Landmark</p>
            <h1 style="margin-top:0;">{top_label}</h1>
            <h3 style="color: #C69C6D;">{top_conf*100:.1f}% Confidence</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Spacer

        # TABS INTERFACE
        tab1, tab2, tab3 = st.tabs(["📊 Analysis", "📜 History", "📍 Location"])

        with tab1:
            st.write("**Top 3 Predictions:**")
            # Get Top 3
            top_3_indices = output_data.argsort()[-3:][::-1]
            top_3_labels = [class_names[i] for i in top_3_indices]
            top_3_scores = [output_data[i] for i in top_3_indices]

            # Custom Progress Bars
            for label, score in zip(top_3_labels, top_3_scores):
                st.write(f"{label}")
                st.progress(float(score))

        with tab2:
            # Fetch Info from Dictionary (or use default)
            info = LANDMARK_INFO.get(top_label, LANDMARK_INFO["Unknown"])
            st.markdown(f"**About {top_label}:**")
            st.write(info.get("desc", "No description available."))
            
        with tab3:
            info = LANDMARK_INFO.get(top_label, LANDMARK_INFO["Unknown"])
            coords = info.get("coords", [21.4858, 39.1925])
            
            # Simple Map
            map_data = pd.DataFrame({'lat': [coords[0]], 'lon': [coords[1]]})
            st.map(map_data, zoom=14)

else:
    # Empty State with a nice helper image
    st.info("👈 Upload an image to begin the analysis.")