import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle

# 1. Page Configuration
st.set_page_config(page_title="Fruit Freshness Classifier", page_icon="🍎", layout="centered")

# Custom layer loader to bypass version mismatch errors
class SafeDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# 2. Re-building the CNN Architecture dynamically to avoid Keras version mismatch
def build_cnn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model

# 3. Load All 4 Saved Files Safely using Cache
@st.cache_resource
def load_all_models():
    # Load CNN Weights from your saved .h5 file
    cnn = build_cnn()
    try:
        cnn.load_weights('cnn_model_fruits.h5')
    except:
        st.error("Missing file: 'cnn_model_fruits.h5' not found in repository.")
    
    # Load SVM Model
    try:
        with open('svm_model.pkl', 'rb') as f:
            svm = pickle.load(f)
    except:
        svm = None
        
    # Load KNN Model
    try:
        with open('knn_model.pkl', 'rb') as f:
            knn = pickle.load(f)
    except:
        knn = None
        
    # Load Label Encoder dynamically
    try:
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
    except:
        le = None
        
    return cnn, svm, knn, le

with st.spinner("Loading AI Models... Please wait"):
    cnn_model, svm_model, knn_model, label_encoder = load_all_models()

# 4. User Interface
st.title("🍎 Fruit Freshness Detection System")
st.write("Upload an image of a fruit to compare predictions between **CNN**, **SVM**, and **KNN** models.")

uploaded_file = st.file_uploader("Choose a fruit image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("---")
    with st.spinner("Analyzing image across all models..."):
        # 5. Image Preprocessing (100x100)
        img_resized = image.convert('RGB').resize((100, 100))
        img_array = np.array(img_resized) / 255.0
        
        # Preprocessing for CNN (Needs batch dimension)
        cnn_input = np.expand_dims(img_array, axis=0)
        
        # Preprocessing for SVM & KNN (Needs flattened vector)
        flat_input = img_array.reshape(1, -1)
        
        # 6. Model Predictions & Decoding via Label Encoder
        # CNN Prediction
        cnn_preds = cnn_model.predict(cnn_input)
        cnn_idx = np.argmax(cnn_preds[0])
        if label_encoder:
            cnn_result = label_encoder.inverse_transform([cnn_idx])[0]
        else:
            cnn_result = "Fresh" if cnn_idx == 0 else "Rotten"
        cnn_conf = cnn_preds[0][cnn_idx] * 100
        
        # SVM Prediction
        if svm_model and label_encoder:
            svm_idx = svm_model.predict(flat_input)[0]
            svm_result = label_encoder.inverse_transform([svm_idx])[0]
        else:
            svm_result = "SVM file missing"
            
        # KNN Prediction
        if knn_model and label_encoder:
            knn_idx = knn_model.predict(flat_input)[0]
            knn_result = label_encoder.inverse_transform([knn_idx])[0]
        else:
            svm_result = "KNN file missing"

    # 7. Displaying Results in Comparison Columns
    st.subheader("🤖 Models Comparison Results:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🧠 CNN Model")
        if "fresh" in cnn_result.lower():
            st.success(f"**{cnn_result}**\n\nConfidence: {cnn_conf:.2f}%")
        else:
            st.error(f"**{cnn_result}**\n\nConfidence: {cnn_conf:.2f}%")
            
    with col2:
        st.markdown("### 📐 SVM Model")
        if "fresh" in svm_result.lower():
            st.success(f"**{svm_result}**")
        elif "rotten" in svm_result.lower():
            st.error(f"**{svm_result}**")
        else:
            st.warning(svm_result)
            
    with col3:
        st.markdown("### 📊 KNN Model")
        if "fresh" in knn_result.lower():
            st.success(f"**{knn_result}**")
        elif "rotten" in knn_result.lower():
            st.error(f"**{knn_result}**")
        else:
            st.warning(knn_result)