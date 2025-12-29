import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Make sure best.pt is in the working directory

# Set up Streamlit interface
st.title("Object Detection App")
st.write("Upload an image to detect Phone Use, Seatbelt, and Smoking.")

# Upload image section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image with PIL
    image = Image.open(uploaded_file)

    # Convert to OpenCV format
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run the model
    results = model(img_bgr)  # This performs inference

    # Display results
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write(results.pandas().xywh)  # Displaying the results in a table format

    # Displaying bounding boxes and labels
    st.write("Detected Objects:")
    results.show()  # This will show the image with bounding boxes
