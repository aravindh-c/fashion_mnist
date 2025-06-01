# app.py - Fashion MNIST Prediction

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("fashion_model.keras")
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define preprocessing function
def preprocess_canvas_image(image_data):
    # Convert RGBA to grayscale
    gray = cv2.cvtColor(image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Threshold
    th = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours to center the object
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        centered = th[y:y+h, x:x+w]
    else:
        centered = th

    # Resize to 18x18
    resized = cv2.resize(centered, (18, 18), interpolation=cv2.INTER_AREA)

    # Pad to 28x28
    padded = np.pad(resized, ((5, 5), (5, 5)), mode='constant', constant_values=0)

    # Normalize and reshape
    padded = padded / 255.0
    return padded.reshape(1, 28, 28, 1)

# Streamlit UI
st.title("Fashion MNIST Classifier")
st.markdown("Draw or upload a fashion item and get its predicted label.")

uploaded_file = st.file_uploader("Upload an image (28x28 grayscale or drawing)", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGBA")
    img = img.resize((280, 280))
    img_np = np.array(img)

    st.image(img_np, caption="Input Image", use_container_width =True)
    processed = preprocess_canvas_image(img_np)
    pred = model.predict(processed)
    label = labels[np.argmax(pred)]
    confidence = np.max(pred) * 100
    st.success(f"Prediction: {label} ({confidence:.2f}%)")
    st.subheader("Class-wise Confidence Scores")
    for lbl, score in zip(labels, pred):
        st.write(f"{lbl}: {float(score) * 100:.2f}%")

    st.subheader("Confidence Bar Chart")
    fig, ax = plt.subplots()
    ax.barh(labels, pred * 100, color='skyblue')
    ax.set_xlabel("Confidence (%)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    st.pyplot(fig)
