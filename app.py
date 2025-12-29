import streamlit as st
from PIL import Image
import torch
import numpy as np

# Function to load the trained YOLOv8 model
@st.cache(allow_output_mutation=True)
def load_model():
    # Load YOLOv8 model from best.pt file
    model = torch.hub.load('ultralytics/yolov8', 'custom', path='best.pt')  
    return model

# Load the model once
model = load_model()

# Function to perform object detection
def detect_objects(model, img):
    # Perform detection
    results = model(img)
    return results

# Streamlit Web Interface
st.title("Driver Distraction Detection")
st.write("Upload an image to detect Phone Use, Seatbelt, and Smoking.")

# File uploader to get the image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    # Convert image to NumPy array for YOLO model processing
    img_array = np.array(img)

    # Perform detection
    st.write("Processing the image...")
    results = detect_objects(model, img_array)

    # Display detection results in tabular format
    st.write("Detection Results:")
    st.write(results.pandas().xywh)  # Display bounding boxes and labels
    
    # Show the image with bounding boxes
    st.image(results.render()[0], caption="Detected Image with Bounding Boxes", use_column_width=True)
