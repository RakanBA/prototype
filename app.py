import streamlit as st
import tensorflow as tf
import requests
import os
import numpy as np
from PIL import Image, ImageOps

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Hugging Face Link Tester", page_icon="📡")
st.title("📡 Cloud Asset Tester")
st.caption("Verifying Hugging Face Direct Links...")

# YOUR HUGGING FACE URLS
MODEL_URL = "https://huggingface.co/RakanBA/heritage-vision-v1/resolve/main/master_classifier.tflite?download=true"
LABELS_URL = "https://huggingface.co/RakanBA/heritage-vision-v1/resolve/main/labels.txt?download=true"

# Local filenames to save them as
LOCAL_MODEL = "downloaded_model.tflite"
LOCAL_LABELS = "downloaded_labels.txt"

# ==========================================
# 2. DOWNLOADER FUNCTION
# ==========================================
def download_file(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Check for 404/403 errors
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True, "Success"
    except Exception as e:
        return False, str(e)

# ==========================================
# 3. TEST LOGIC
# ==========================================
# A. Download Labels
with st.status("Downloading Assets from Cloud...", expanded=True) as status:
    st.write("1. Fetching Labels...")
    success, msg = download_file(LABELS_URL, LOCAL_LABELS)
    if success:
        st.write("✅ Labels downloaded.")
    else:
        st.error(f"❌ Failed to download labels: {msg}")
        st.stop()
        
    st.write("2. Fetching TFLite Model (25MB)...")
    success, msg = download_file(MODEL_URL, LOCAL_MODEL)
    if success:
        st.write("✅ Model downloaded.")
    else:
        st.error(f"❌ Failed to download model: {msg}")
        st.stop()
    
    status.update(label="Assets Ready!", state="complete", expanded=False)

# ==========================================
# 4. LOAD & RUN (Sanity Check)
# ==========================================
try:
    # Load Labels
    with open(LOCAL_LABELS, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    st.success(f"Loaded {len(class_names)} Classes from Cloud Text File.")

    # Load Model (Using TFLite Interpreter)
    interpreter = tf.lite.Interpreter(model_path=LOCAL_MODEL)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.success("TFLite Model Loaded Successfully!")
    
except Exception as e:
    st.error(f"❌ File is corrupted or invalid: {e}")
    st.stop()

# ==========================================
# 5. LIVE PREDICTION
# ==========================================
st.divider()
st.subheader("Test the Cloud Brain")

uploaded_file = st.file_uploader("Upload an image to verify inference:", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Preprocessing (Must match your training!)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=200)
    
    # Resize & Normalize
    img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    input_data = np.array(img_resized, dtype=np.float32)
    input_data = input_data / 255.0  # Normalize 0-1
    input_data = np.expand_dims(input_data, axis=0) # Add batch dimension [1, 224, 224, 3]

    # Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get Result
    index = np.argmax(output_data)
    confidence = output_data[0][index]
    label = class_names[index]
    
    # Display
    st.info(f"Prediction: **{label}**")
    st.caption(f"Confidence: {confidence*100:.2f}%")