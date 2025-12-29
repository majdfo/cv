import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T
from io import BytesIO

# Load the model
model = torch.load('best.pt')
model.eval()  # Set the model to evaluation mode

# Define the transformation (assuming your model uses common transforms like Resize, ToTensor)
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to predict on the image
def predict_image(image):
    # Apply transformations to the uploaded image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Run the model to get predictions
    with torch.no_grad():
        outputs = model(image)
    
    # Process the outputs (modify as needed for your model's output format)
    # Assuming the model outputs a dictionary with class labels and scores
    predictions = outputs[0]  # Adjust based on how your model outputs
    return predictions

# Streamlit Interface
def main():
    st.title("Phone Use, Seatbelt, and Smoking Detection")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        
        # Make prediction
        predictions = predict_image(image)
        
        # Display results
        st.write("Prediction Results:")
        st.write("Phone Use:", predictions['PhoneUse'])
        st.write("Seatbelt:", predictions['Seatbelt'])
        st.write("Smoking:", predictions['Smoking'])

if __name__ == '__main__':
    main()
