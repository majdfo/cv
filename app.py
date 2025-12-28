import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

# 1. تحميل الموديل
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 2. دالة التنبؤ والرسم
def predict_and_draw(image_array):
    results = model(image_array)  # الحصول على نتائج من النموذج
    img = image_array.copy()

    for result in results:  # التعامل مع جميع النتائج (في حالة وجود أكثر من فئة)
        boxes = result.boxes  # الحصول على الصناديق
        
        for box in boxes:
            conf = float(box.conf[0])  # نسبة الثقة
            
            if conf > 0.25:  # شرط الثقة
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # استخراج الإحداثيات
                label = model.names[int(box.cls[0])]  # استخراج اسم الفئة
                
                # تحديد اللون بناءً على الفئة
                color = (0, 255, 0) if label == "PhoneUse" else (255, 0, 0) if label == "Seatbelt" else (0, 0, 255)
                
                # رسم المستطيل
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                
                # كتابة النص
                text = f"{label} {conf:.2f}"
                cv2.putText(img, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return img

# 3. واجهة التطبيق
st.title("Driver Distraction Detection (YOLOv8) 🚗")
st.write("نظام كشف تشتت السائق - متوافق مع Streamlit Cloud")

option = st.radio("اختر طريقة الإدخال:", ("التقاط صورة (كاميرا)", "رفع صورة من الجهاز"))

if option == "التقاط صورة (كاميرا)":
    img_file = st.camera_input("التقط صورة الآن")
    
    if img_file is not None:
        image = Image.open(img_file)
        img_array = np.array(image)
        
        # المعالجة
        res_img = predict_and_draw(img_array)
        
        # العرض
        st.image(res_img, caption="النتيجة", use_column_width=True)

elif option == "رفع صورة من الجهاز":
    uploaded_file = st.file_uploader("اختر صورة...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة الأصلية", use_column_width=True)
        
        img_array = np.array(image)
        
        # المعالجة
        res_img = predict_and_draw(img_array)
        
        # العرض
        st.image(res_img, caption="الصورة المعالجة", use_column_width=True)
