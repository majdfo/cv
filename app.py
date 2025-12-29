import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO  # استيراد YOLO من ultralytics

# محاولة تحميل النموذج باستخدام YOLOv8
try:
    # استخدام الطريقة الرسمية لتحميل النموذج باستخدام YOLOv8
    model = YOLO('best.pt')  # تأكد من أن النموذج في نفس المجلد أو قدم المسار الصحيح
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Streamlit Interface
def main():
    st.title("Phone Use, Seatbelt, and Smoking Detection")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        
        # تحويل الصورة إلى المسار الصحيح لتتمكن YOLO من معالجتها
        image_path = uploaded_file.name
        
        try:
            # Make prediction using the model
            results = model(image_path)
            
            # عرض النتائج
            st.write("Prediction Results:")
            st.write(f"Detected {len(results)} objects in the image.")
            results.show()  # ستعرض الصورة مع المربعات المحددة للتنبؤات
        except Exception as e:
            st.write(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
