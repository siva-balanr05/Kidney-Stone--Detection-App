import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# Define paths
MODEL_PATH = r"C:\ML_Model\kidney_model.keras"
    
# Load trained model
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_trained_model()

# Define class labels (Ensure they match training dataset)
class_labels = ["Normal", "Tumor", "Cyst"]

# Streamlit UI
st.title("Kidney Stone Detection App 🏥")
st.write("Upload an image to detect if it is **Normal, Tumor, or Cyst**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale image
    return img_array

# If an image is uploaded, process and predict
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing image...")

    # Preprocess the image
    processed_image = preprocess_image(uploaded_file)

    # Make a prediction
    predictions = model.predict(processed_image)

    # Get the predicted class and probability
    predicted_class = class_labels[np.argmax(predictions)]
    prediction_probability = np.max(predictions) * 100

    # Display the prediction and confidence
    st.write(f"### Prediction: **{predicted_class}**")
    st.write(f"### Confidence: **{prediction_probability:.2f}%**")

    # Provide recommendations
    if predicted_class == "Normal":
        st.success("✅ The image appears to be normal.")
    elif predicted_class == "Tumor":
        st.warning("⚠️ Ablation techniques (cryoablation or radiofrequency ablation) for inoperable tumors.")
    elif predicted_class == "Cyst":
        st.warning("⚠️ Aspiration + sclerotherapy (injecting alcohol to shrink the cyst) /Laparoscopic cyst decortication (surgical removal).")
