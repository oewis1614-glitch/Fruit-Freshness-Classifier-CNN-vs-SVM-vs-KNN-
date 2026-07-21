import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pickle
import urllib.request

# 1. Page Configuration
st.set_page_config(page_title="Fruit Freshness Comparison", page_icon="🍎", layout="centered")

# 2. Dynamic CNN Architecture
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

# Helper function to download files from Google Drive
def download_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        download_url = f'https://docs.google.com/uc?export=download&id={file_id}'
        urllib.request.urlretrieve(download_url, output_path)

# 3. Dynamic Cache and Loader for ALL 4 Assets with your Google Drive IDs
@st.cache_resource
def load_all_assets():
    # ה-IDs الخاصة بملفاتك على Google Drive
    CNN_ID = '1GeWbPgdgeFcwxuAlhifPGHZiUnzUoFy8'
    SVM_ID = '1D_9HGpqu5DKjLbNKN2elQ6nvef3imSYL'
    KNN_ID = '10OH7pvx7Agu8UNYxAs31-tl1VhxvwnSu'
    LE_ID  = '1-nc7fp9KfsCfCOxznjy0yuXEwLnSk_Br'
    
    with st.spinner("Downloading AI Models from Google Drive... Please wait (Only happens once)"):
        download_from_drive(CNN_ID, 'cnn_model_fruits.h5')
        download_from_drive(SVM_ID, 'svm_model.pkl')
        download_from_drive(KNN_ID, 'knn_model.pkl')
        download_from_drive(LE_ID, 'label_encoder.pkl')
        
    # Load CNN Model
    cnn = build_cnn()
    cnn.load_weights('cnn_model_fruits.h5')
    
    # Load SVM Model
    with open('svm_model.pkl', 'rb') as f:
        svm = pickle.load(f)
        
    # Load KNN Model
    with open('knn_model.pkl', 'rb') as f:
        knn = pickle.load(f)
        
    # Load Label Encoder
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
        
    return cnn, svm, knn, le

cnn_model, svm_model, knn_model, label_encoder = load_all_assets()

# 4. User Interface
st.title("🍎 Fruit Freshness Comparison System")
st.write("Upload a fruit image to evaluate and compare **CNN**, **SVM**, and **KNN** models simultaneously.")

uploaded_file = st.file_uploader("Choose a fruit image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("---")
    with st.spinner("Running predictions across all 3 architectures..."):
        # Preprocessing (100x100)
        img_resized = image.convert('RGB').resize((100, 100))
        img_array = np.array(img_resized) / 255.0
        
        cnn_input = np.expand_dims(img_array, axis=0)
        flat_input = img_array.reshape(1, -1)
        
        # Predictions
        # 1. CNN
        cnn_preds = cnn_model.predict(cnn_input)
        cnn_idx = np.argmax(cnn_preds[0])
        cnn_result = label_encoder.inverse_transform([cnn_idx])[0]
        cnn_conf = cnn_preds[0][cnn_idx] * 100
        
        # 2. SVM
        svm_idx = svm_model.predict(flat_input)[0]
        svm_result = label_encoder.inverse_transform([svm_idx])[0]
        
        # 3. KNN
        knn_idx = knn_model.predict(flat_input)[0]
        knn_result = label_encoder.inverse_transform([knn_idx])[0]

    # Displaying Results Side-by-Side
    st.subheader("🤖 Comprehensive Evaluation Matrix:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🧠 CNN Model")
        if "fresh" in str(cnn_result).lower():
            st.success(f"**{cnn_result}**\n\nConfidence: {cnn_conf:.2f}%")
        else:
            st.error(f"**{cnn_result}**\n\nConfidence: {cnn_conf:.2f}%")
            
    with col2:
        st.markdown("### 📐 SVM Model")
        if "fresh" in str(svm_result).lower():
            st.success(f"**{svm_result}**")
        else:
            st.error(f"**{svm_result}**")
            
    with col3:
        st.markdown("### 📊 KNN Model")
        if "fresh" in str(knn_result).lower():
            st.success(f"**{knn_result}**")
        else:
            st.error(f"**{knn_result}**")
