import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import time

# 1. تحميل نموذج YOLO
@st.cache_resource
def load_model():
    model = YOLO('best.pt')
    return model

model = load_model()

# 2. دالة التنبؤ والرسم (تم تعديلها لتتوافق مع YOLOv8)
def predict_and_draw(frame):
    # إجراء التنبؤ
    results = model(frame)
    
    img = frame.copy()
    
    # التعامل مع النتائج بطريقة YOLOv8 الصحيحة
    for result in results:
        boxes = result.boxes  # الوصول للصناديق (Boxes)
        for box in boxes:
            conf = box.conf[0]
            if conf > 0.25:  # شرط الثقة
                # استخراج الإحداثيات (تحويلها لأرقام صحيحة)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # استخراج اسم الكلاس
                cls = int(box.cls[0])
                label = model.names[cls]
                
                # تحديد اللون بناءً على الحالة
                if label == "PhoneUse":
                    color = (0, 255, 0)  # أخضر
                elif label == "Seatbelt":
                    color = (255, 0, 0)  # أزرق (لأن OpenCV يستخدم BGR)
                else:
                    color = (0, 0, 255)  # أحمر
                
                # رسم المستطيل والنص
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return img

# 3. دالة الكاميرا (ملاحظة: cv2.VideoCapture لا يعمل على Streamlit Cloud)
def open_camera_and_detect():
    st.warning("⚠️ تنبيه: الكاميرا المباشرة (Live Camera) لا تعمل عادةً على استضافة Streamlit Cloud المجانية. يفضل استخدام خيار 'Upload Image'.")
    
    # نستخدم camera_input لأنه الوحيد المدعوم على السحابة
    img_file = st.camera_input("التقط صورة")
    
    if img_file is not None:
        image = Image.open(img_file)
        image_array = np.array(image)
        result_img = predict_and_draw(image_array)
        st.image(result_img, caption="النتيجة", use_column_width=True)

# 4. رفع صورة
def upload_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        
        # تحويل الصورة إلى مصفوفة NumPy
        image_array = np.array(image)
        
        # تطبيق الكشف والرسم
        result_img = predict_and_draw(image_array)
        
        # تحويل النتيجة لعرضها
        result_image = Image.fromarray(result_img)
        
        st.image(result_image, caption="Processed Image with Bounding Boxes.", use_column_width=True)
        
        return result_image
    return None

# 5. تحميل الصورة المعدلة
def download_image(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    st.download_button(
        label="Download Processed Image",
        data=img_byte_arr,
        file_name="processed_image.png",
        mime="image/png"
    )

# --- واجهة التطبيق الرئيسية ---
st.title("Driver Distraction Detection")
st.subheader("YOLOv8 Detection App")

# خيارات للمستخدم
option = st.radio("Choose an option", ("Use Camera (Mobile/Webcam)", "Upload Image"))

if option == "Use Camera (Mobile/Webcam)":
    open_camera_and_detect()

elif option == "Upload Image":
    image = upload_image()
    if image is not None:
        download_image(image)
