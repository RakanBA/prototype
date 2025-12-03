import streamlit as st
import tensorflow as tf
import requests
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(
    page_title="Heritage Vision AI",
    page_icon="🏛️",
    layout="wide",  # Uses the full width of the screen
    initial_sidebar_state="expanded"
)

# Custom CSS to hide default Streamlit clutter and make it look clean
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    div.stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. CONSTANTS & CACHING
# ==========================================
MODEL_URL = "https://huggingface.co/RakanBA/heritage-vision-v1/resolve/main/master_classifier.tflite?download=true"
LABELS_URL = "https://huggingface.co/RakanBA/heritage-vision-v1/resolve/main/labels.txt?download=true"
LOCAL_MODEL = "master_classifier.tflite"
LOCAL_LABELS = "labels.txt"

@st.cache_resource(show_spinner="Downloading Model Components...")
def load_assets():
    """
    Downloads and loads the model assets only ONCE.
    Detailed logging helps debug connection issues.
    """
    # 1. Download Labels
    if not os.path.exists(LOCAL_LABELS):
        try:
            r = requests.get(LABELS_URL)
            r.raise_for_status()
            with open(LOCAL_LABELS, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            st.error(f"CRITICAL ERROR: Could not download labels.\n{e}")
            st.stop()
            
    # 2. Download Model
    if not os.path.exists(LOCAL_MODEL):
        try:
            r = requests.get(MODEL_URL)
            r.raise_for_status()
            with open(LOCAL_MODEL, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            st.error(f"CRITICAL ERROR: Could not download model.\n{e}")
            st.stop()

    # 3. Load into Memory
    try:
        with open(LOCAL_LABELS, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
            
        interpreter = tf.lite.Interpreter(model_path=LOCAL_MODEL)
        interpreter.allocate_tensors()
        
        return interpreter, classes
    except Exception as e:
        st.error(f"Model File Corrupted: {e}")
        st.stop()

# Load assets immediately on app start
interpreter, class_names = load_assets()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3067/3067451.png", width=100)
    st.title("Heritage Vision")
    st.info("This AI system identifies historical landmarks in Old Jeddah using Convolutional Neural Networks.")
    st.divider()
    st.write("### ⚙️ Model Specs")
    st.write(f"- **Type:** TFLite (Quantized)")
    st.write(f"- **Classes:** {len(class_names)}")
    st.write(f"- **Input Shape:** {input_details[0]['shape']}")

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================
st.title("🏛️ Landmark Recognition System")
st.write("Upload a photo of a historical site to analyze it.")

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Create two columns for a better layout
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.subheader("Your Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True, caption="Uploaded Photo")

    with col2:
        st.subheader("AI Analysis")
        
        with st.spinner("Analyzing pixels..."):
            # Prepare Image
            img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
            input_data = np.array(img_resized, dtype=np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            # Run Inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            # Get Top Prediction
            top_index = np.argmax(output_data)
            top_label = class_names[top_index]
            top_conf = output_data[top_index]

        # 1. Display Big Result
        if top_conf > 0.7:
            st.success(f"**Identified: {top_label}**")
        else:
            st.warning(f"**Uncertain: {top_label}** (Low Confidence)")
            
        st.metric(label="Confidence Score", value=f"{top_conf*100:.1f}%")

        # 2. Confidence Chart (The Plot You Asked For)
        st.write("### 📊 Confidence Distribution")
        
        # Create a clean DataFrame for the chart
        chart_data = pd.DataFrame({
            "Landmark": class_names,
            "Confidence": output_data
        }).sort_values(by="Confidence", ascending=False).head(5) # Show top 5 only
        
        # Display interactive bar chart
        st.bar_chart(
            chart_data, 
            x="Landmark", 
            y="Confidence", 
            color="#FF4B4B", # Matches the button style
            use_container_width=True
        )

else:
    # Placeholder when no image is uploaded
    st.info("👈 Waiting for upload...")