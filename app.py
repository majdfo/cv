import streamlit as st
import torch
from PIL import Image
import numpy as np
from my_utils import load_model, detect_objects


# تحميل النموذج المدرب
model = load_model('best.pt')

def main():
    st.title("Detection of Driver Distractions")
    st.write("Upload an image to detect Phone Use, Seatbelt, and Smoking.")

    # رفع الصورة
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image.", use_column_width=True)

        # معالجة الصورة
        st.write("Processing the image...")
        img = np.array(img)  # تحويل الصورة إلى مصفوفة نمرية

        # تنفيذ الكشف باستخدام النموذج
        results = detect_objects(model, img)

        # عرض النتيجة
        st.write("Detection Results:")
        st.image(results, caption="Detection Results", use_column_width=True)

if __name__ == "__main__":
    main()
