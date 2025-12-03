import streamlit as st
import tensorflow as tf
import requests
import os
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image, ImageOps

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Heritage Vision AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Standard Clean CSS (No colors, just spacing)
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    div.stButton > button { width: 100%; border-radius: 5px; }
    
    .result-card {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e6e6e6;
        background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA: HISTORICAL INFO (MOCK DB)
# ==========================================
# Ensure these keys match your 'labels.txt' exactly
LANDMARK_INFO = {
    "Nasseef House": {
        "desc": "Built in 1881, Nasseef House is historically significant as the residence of King Abdulaziz when he entered Jeddah in 1925.",
        "location": "Old Jeddah (Al-Balad)",
        "coords": [21.4833, 39.1833] 
    },
    "Al-Alawi Souq": {
        "desc": "One of the oldest markets in the region, connecting the port to the Makkah Gate. Famous for spices and incense.",
        "location": "Heart of Al-Balad",
        "coords": [21.4850, 39.1870]
    },
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
    st.title("Heritage Vision")
    st.info("System Status: Online 🟢")
    st.markdown("---")
    st.write("### 📂 Project Details")
    st.caption("This prototype demonstrates CNN-based classification for historical preservation.")

# ==========================================
# 5. MAIN INTERFACE
# ==========================================
st.title("Landmark Recognition System")
st.write("Upload a photo of a historical site in Al-Balad to identify it.")

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # --- PROCESSING ---
    image = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner("Analyzing image..."):
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
        st.subheader("Input Image")
        st.image(image, use_container_width=True)
        
        if top_conf > 0.85:
            st.success("High Confidence Match")

    with col2:
        # Result Header
        st.subheader("Analysis Results")
        
        # Clean Card Design
        st.markdown(f"""
        <div class="result-card">
            <h2 style="margin:0; color: #333;">{top_label}</h2>
            <p style="margin:0; color: #666;">Confidence: <b>{top_conf*100:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Spacer

        # TABS INTERFACE
        tab1, tab2, tab3 = st.tabs(["📊 Confidence", "📜 History", "📍 Location"])

        with tab1:
            st.write("**Prediction Distribution**")
            
            # Interactive Chart
            top_5_indices = output_data.argsort()[-5:][::-1]
            top_5_labels = [class_names[i] for i in top_5_indices]
            top_5_scores = [output_data[i] for i in top_5_indices]
            
            df_chart = pd.DataFrame({"Landmark": top_5_labels, "Probability": top_5_scores})
            
            fig = px.bar(
                df_chart, 
                x="Probability", 
                y="Landmark", 
                orientation='h',
                text_auto='.1%',
                color="Probability",
                color_continuous_scale="Reds" # Standard Red scale
            )
            fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0))
            fig.update_xaxes(visible=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Fetch Info
            info = LANDMARK_INFO.get(top_label, LANDMARK_INFO["Unknown"])
            st.markdown(f"**About {top_label}**")
            st.write(info.get("desc", "No description available."))
            
        with tab3:
            info = LANDMARK_INFO.get(top_label, LANDMARK_INFO["Unknown"])
            coords = info.get("coords", [21.4858, 39.1925])
            
            # Simple Map
            map_data = pd.DataFrame({'lat': [coords[0]], 'lon': [coords[1]]})
            st.map(map_data, zoom=15)

else:
    st.info("👈 Waiting for image upload...")