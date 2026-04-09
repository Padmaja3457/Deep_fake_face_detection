from os import path, remove
from random import randint
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from model import get_model, get_model_summary, model_prediction

# Function for face segmentation
def identify_fake_parts(image_path, threshold):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fake_mask = np.zeros_like(binary)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < threshold:
            cv2.drawContours(fake_mask, [contour], -1, (255), thickness=cv2.FILLED)
    return fake_mask

# Streamlit app setup
st.set_page_config(page_title="Real Fake Face Classification",
                   page_icon=path.join('static', 'icons', 'logo.png'))

with open(path.join('static', 'styles.css')) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title('Real Fake Face Image Classifier')

st.markdown(
    """
    This real-fake face classifier utilizes an InceptionResNetV2-based convolutional neural network. 
    Trained on a Kaggle dataset comprising 140,000 images (70,000 real and 70,000 fake), the classifier achieves an accuracy of 93%. 
"""
)

# Load Model
finalmodel_path = path.join('static', 'finalmodel', 'final_model_fixed.h5')
finalmodel = get_model(finalmodel_path)

uploaded_file = st.file_uploader('Upload a face image for prediction',
                                 type=['png', 'jpg', 'jpeg'],
                                 accept_multiple_files=False)

threshold = st.slider('Segmentation Threshold', min_value=100, max_value=5000, value=1000)

# Logical block for file handling
if uploaded_file is not None:
    uploaded_file_name = uploaded_file.name
    with open(uploaded_file_name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    fake_mask = identify_fake_parts(uploaded_file_name, threshold)
    prediction_label, real_face_prob, fake_face_prob = model_prediction(uploaded_file_name, finalmodel)
    remove(uploaded_file_name) 
else:
    embedded_img = path.join('static', 'faces', f'{randint(1, 6)}.jpg')
    st.image(Image.open(embedded_img).resize((720, 720)))
    fake_mask = identify_fake_parts(embedded_img, threshold)
    prediction_label, real_face_prob, fake_face_prob = model_prediction(embedded_img, finalmodel)

# --- CORRECTED OUTPUT LOGIC STARTS HERE ---

# 1. Show Segmented Image ONLY if it's Fake
if prediction_label == 'Fake Face':
    with st.expander(label='Segmented Image', expanded=False):
        st.image(fake_mask, caption='Identified Fake Parts (Expanded)', use_container_width=True)

# 2. Show Probabilities for BOTH Real and Fake (Move this OUT of any if-block)
with st.expander(label='Prediction Probabilities', expanded=True):
    st.markdown('<h3>Results</h3>', unsafe_allow_html=True)

    row_11, row_12 = st.columns([1, 1])
    row_11.markdown('<h4>Face Type</h4>', unsafe_allow_html=True)
    row_12.markdown('<h5>Confidence</h5>', unsafe_allow_html=True)

    # REAL Progress Row
    row_21, row_22 = st.columns([1, 1])
    row_21.markdown('<h3>REAL</h3>', unsafe_allow_html=True)
    row_22.markdown(f'<h2>{real_face_prob}%</h2>', unsafe_allow_html=True)
    st.progress(real_face_prob / 100)

    # FAKE Progress Row
    row_31, row_32 = st.columns([1, 1])
    row_31.markdown('<h3>FAKE</h3>', unsafe_allow_html=True)
    row_32.markdown(f'<h2>{fake_face_prob}%</h2>', unsafe_allow_html=True)
    st.progress(fake_face_prob / 100)

# Final Prediction Text
st.markdown(f"The classifier's prediction is that the loaded image is a **{prediction_label}**❗")