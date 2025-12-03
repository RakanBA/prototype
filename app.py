import streamlit as st
import tensorflow as tf
import requests
import os
import numpy as np
import pandas as pd
import plotly.express as px  # NEW: For professional charts
from PIL import Image, ImageOps

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Heritage Vision AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div.stButton > button { width: 100%; border-radius: 5px; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOAD ASSETS (Cached)
# ==========================================
MODEL_URL = "https://huggingface.co/RakanBA/heritage-vision-v1/resolve/main/master_classifier.tflite?download=true"
LABELS_URL = "https://huggingface.co/RakanBA/heritage-vision-v1/resolve/main/labels.txt?download=true"
LOCAL_MODEL = "master_classifier.tflite"
LOCAL_LABELS = "labels.txt"

@st.cache_resource
def load_assets():
    # Download logic (Same as before, just condensed)
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
# 3. SIDEBAR (Context & Info)
# ==========================================
with st.sidebar:
    st.title("🏛️ Heritage Vision")
    st.markdown("---")
    st.info("**About:** This system uses a CNN (Convolutional Neural Network) to classify historical landmarks in Old Jeddah.")
    
    st.write("### 🛠️ How it works")
    st.markdown("""
    1. Image is resized to **224x224**.
    2. Normalized to range **[0,1]**.
    3. Processed by **TFLite Quantized Model**.
    """)
    st.markdown("---")
    st.caption(f"v1.0 | Classes: {len(class_names)}")

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
st.title("Landmark Recognition System")
st.markdown("##### 📸 Upload a photo of a historical site to identify it.")

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1.5], gap="large")

    # --- LEFT COLUMN: IMAGE ---
    with col1:
        st.subheader("Input Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True, caption="Source Image")

    # --- RIGHT COLUMN: ANALYTICS ---
    with col2:
        st.subheader("AI Diagnosis")
        
        with st.spinner("Processing..."):
            # Preprocessing
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

        # 1. Primary Result Card
        result_color = "green" if top_conf > 0.7 else "orange"
        result_icon = "✅" if top_conf > 0.7 else "⚠️"
        
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid {result_color};">
            <h2 style="margin:0; color: #333;">{result_icon} {top_label}</h2>
            <p style="margin:0; color: #666;">Confidence Score: <b>{top_conf*100:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Spacer

        # 2. Interactive Probability Chart (Plotly)
        st.markdown("### 📊 Probability Breakdown")
        
        # Prepare data for top 5
        # Get indices of top 5
        top_5_indices = output_data.argsort()[-5:][::-1]
        top_5_labels = [class_names[i] for i in top_5_indices]
        top_5_scores = [output_data[i] for i in top_5_indices]
        
        df_chart = pd.DataFrame({
            "Landmark": top_5_labels,
            "Probability": top_5_scores
        })
        
        # Plotly Bar Chart (Much nicer than default)
        fig = px.bar(
            df_chart, 
            x="Probability", 
            y="Landmark", 
            orientation='h',
            text_auto='.1%',
            color="Probability",
            color_continuous_scale="Reds"
        )
        fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(visible=False) # Hide x axis numbers for cleaner look
        st.plotly_chart(fig, use_container_width=True)

        # 3. Feedback Loop (Simulated)
        with st.expander("Is this result incorrect?"):
            st.write("Help us improve the model:")
            c1, c2 = st.columns(2)
            if c1.button("❌ Wrong Landmark"):
                st.toast("Feedback recorded! We will review this image.", icon="📝")
            if c2.button("✅ Correct"):
                st.toast("Thanks for verifying!", icon="👍")

else:
    # Empty State
    st.info("👈 Waiting for upload...")
    st.markdown("Try uploading a clear photo of **Nasseef House** or **Al-Alawi Souq**.")