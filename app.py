import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

# 1. تحميل الموديل
@st.cache_resource
def load_model():
    # تأكد أن ملف best.pt موجود بجانب app.py
    return YOLO('best.pt')

model = load_model()

# 2. دالة التنبؤ والرسم (معدلة لتعمل مع YOLOv8)
def predict_and_draw(image_array):
    # YOLOv8 يرجع قائمة من النتائج
    results = model(image_array)
    
    img = image_array.copy()
    
    # التعامل مع النتائج
    for result in results:
        # في YOLOv8، الصناديق موجودة داخل result.boxes
        boxes = result.boxes
        
        for box in boxes:
            # استخراج نسبة الثقة
            conf = float(box.conf[0])
            
            if conf > 0.25:  # شرط الثقة
                # استخراج الإحداثيات وتحويلها لأرقام صحيحة
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # استخراج اسم الكلاس
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                
                # منطق الألوان الخاص بك
                if label == "PhoneUse":
                    color = (0, 255, 0)  # أخضر
                elif label == "Seatbelt":
                    color = (255, 0, 0)  # أزرق (لأن OpenCV يستخدم BGR أحياناً، لكن Streamlit يحب RGB)
                    # للتصحيح: في RGB (أحمر=255, 0, 0)
                else:
                    color = (0, 0, 255)  # أحمر/أزرق حسب التنسيق
                
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
    # هذا الأمر هو الوحيد الذي يعمل على السيرفرات
    img_file = st.camera_input("التقط صورة الآن")
    
    if img_file is not None:
        image = Image.open(img_file)
        # تحويل الصورة إلى مصفوفة NumPy
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
        
        # تحويل الصورة
        img_array = np.array(image)
        
        # المعالجة
        res_img = predict_and_draw(img_array)
        
        # العرض
        st.image(res_img, caption="الصورة المعالجة", use_column_width=True)
