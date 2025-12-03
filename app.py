import streamlit as st
import tensorflow as tf
import requests
import os
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image, ImageOps

# ==========================================
# 1. PAGE CONFIG & ARABIC SUPPORT
# ==========================================
st.set_page_config(
    page_title="Heritage Vision AI",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Arabic Support (RTL) and Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Tajawal', sans-serif;
    }
    
    .rtl-text {
        direction: rtl; 
        text-align: right;
        font-family: 'Tajawal', sans-serif;
    }
    
    .result-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #fdfbf7;
        border: 1px solid #d4c5a9;
        margin-bottom: 20px;
        text-align: center;
    }
    
    div.stButton > button {
        background-color: #8D6E63;
        color: white;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOAD DATA & ASSETS
# ==========================================
MODEL_URL = "https://huggingface.co/RakanBA/heritage-vision-v1/resolve/main/master_classifier.tflite?download=true"
LABELS_URL = "https://huggingface.co/RakanBA/heritage-vision-v1/resolve/main/labels.txt?download=true"
LOCAL_MODEL = "master_classifier.tflite"
LOCAL_LABELS = "labels.txt"
CSV_FILE = "landmarks.csv"

@st.cache_data
def load_landmark_data():
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            df['Site Name'] = df['Site Name'].astype(str).str.strip()
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return None
    return None

@st.cache_resource
def load_ai_model():
    if not os.path.exists(LOCAL_LABELS):
        with open(LOCAL_LABELS, 'wb') as f: f.write(requests.get(LABELS_URL).content)
    if not os.path.exists(LOCAL_MODEL):
        with open(LOCAL_MODEL, 'wb') as f: f.write(requests.get(MODEL_URL).content)

    with open(LOCAL_LABELS, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    interpreter = tf.lite.Interpreter(model_path=LOCAL_MODEL)
    interpreter.allocate_tensors()
    return interpreter, classes

df_landmarks = load_landmark_data()
interpreter, class_names = load_ai_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5110/5110594.png", width=80)
    st.title("Heritage Vision")
    st.markdown("---")
    if df_landmarks is not None:
        st.success(f"📚 Database Loaded: {len(df_landmarks)} Sites")
    
    st.write("### ℹ️ About")
    st.caption("AI-Powered recognition for Al-Balad Historical District in Jeddah.")

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🏛️ Landmark Recognition")
    st.write("Upload a photo to discover the history of Old Jeddah.")

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # --- PROCESSING ---
    image = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner("Analyzing Architecture..."):
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

    # --- RETRIEVE INFO FROM CSV ---
    info = None
    if df_landmarks is not None:
        match = df_landmarks[df_landmarks['Site Name'] == top_label]
        if not match.empty:
            info = match.iloc[0]

    # --- UI LAYOUT ---
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.image(image, use_container_width=True, caption="Uploaded Image")
        if top_conf > 0.85:
            st.success(f"Confidence: {top_conf*100:.1f}%")

    with col2:
        # Styled Result Card
        st.markdown(f"""
        <div class="result-card">
            <h3 style="margin:0; color: #5D4037;">Detected Landmark</h3>
            <h1 style="margin:0; color: #3E2723;">{top_label}</h1>
        </div>
        """, unsafe_allow_html=True)

        # TABS
        tabs = st.tabs(["📜 Info", "🏛️ History & Art", "📍 Map", "📊 AI Stats"])

        # TAB 1: General Info (RTL Text)
        with tabs[0]:
            if info is not None:
                st.markdown(f"""
                <div class="rtl-text">
                <h3>📌 الوصف (Description)</h3>
                <p>{info['Description']}</p>
                <hr>
                <h4>💡 الأهمية (Significance)</h4>
                <p>{info['Site Significance']}</p>
                <h4>🎒 الأنشطة (Activities)</h4>
                <p>{info['Available Activities']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No information available in database.")

        # TAB 2: History & Architecture
        with tabs[1]:
            if info is not None:
                st.markdown(f"""
                <div class="rtl-text">
                <h3>🏺 التاريخ (History)</h3>
                <p>{info['History']}</p>
                <hr>
                <h3>🏗️ العمارة والتصميم (Architecture)</h3>
                <p>{info['Architecture & Design']}</p>
                <br>
                <p><b>⏰ أفضل وقت للزيارة:</b> {info['Best Visiting Times']}</p>
                </div>
                """, unsafe_allow_html=True)

        # TAB 3: Map
        with tabs[2]:
            if info is not None and pd.notnull(info['Latitude']):
                st.map(pd.DataFrame({'lat': [info['Latitude']], 'lon': [info['Longitude']]}))
                st.markdown(f"<div class='rtl-text'>📍 <b>الموقع:</b> {info['Location']}</div>", unsafe_allow_html=True)
            else:
                st.info("Location coordinates missing.")

        # TAB 4: Statistics
        with tabs[3]:
            # Chart
            top_5_indices = output_data.argsort()[-5:][::-1]
            top_5_labels = [class_names[i] for i in top_5_indices]
            top_5_scores = [output_data[i] for i in top_5_indices]
            
            df_chart = pd.DataFrame({"Landmark": top_5_labels, "Probability": top_5_scores})
            fig = px.bar(df_chart, x="Probability", y="Landmark", orientation='h', text_auto='.1%', color="Probability", color_continuous_scale="Oranges")
            fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0))
            fig.update_xaxes(visible=False)
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Please upload an image to start.")