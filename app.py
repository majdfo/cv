import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import time

# تحميل نموذج YOLO
@st.cache_resource
def load_model():
    # تأكد أن ملف best.pt موجود في نفس المجلد
    model = YOLO('best.pt')
    return model

model = load_model()

# دالة لتحديد سبب تشدد السائق ورسم المستطيلات (معدلة لـ YOLOv8)
def predict_and_draw(frame):
    # إجراء التنبؤ
    results = model(frame)
    
    img = frame.copy()
    
    # التعامل مع النتائج في YOLOv8
    for result in results:
        boxes = result.boxes  # الوصول للصناديق
        for box in boxes:
            conf = box.conf[0]
            if conf > 0.25:  # شرط الثقة
                # استخراج الإحداثيات
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # استخراج الصنف (Class)
                cls = int(box.cls[0])
                label = model.names[cls]
                
                # منطق الألوان الخاص بك
                if label == "PhoneUse":
                    color = (0, 255, 0)  # أخضر
                elif label == "Seatbelt":
                    color = (255, 0, 0)  # أزرق (في BGR) أو أحمر حسب الترتيب
                else:
                    color = (0, 0, 255)  # أحمر
                
                # رسم المستطيل والنص
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return img

# دالة لفتح الكاميرا ورسم المستطيلات في الوقت الحقيقي
# ملاحظة: هذا يعمل فقط Localhost ولن يعمل على Streamlit Cloud
def open_camera_and_detect():
    st.text("Opening camera... (Works on Localhost Only)")
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        st.error("Unable to access camera. If on Cloud, use 'Upload Image'.")
        return

    stop_button = st.button('Stop Camera')
    frame_placeholder = st.empty()  # مكان مخصص لعرض الفيديو بسلاسة

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        # كشف الكائنات ورسم المستطيلات
        frame_with_boxes = predict_and_draw(frame)
        
        # تحويل الصورة من BGR إلى RGB للعرض الصحيح
        frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        
        # عرض الصورة في المكان المحجوز
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        if stop_button:
            cap.release()
            st.success("Camera stopped.")
            break

    cap.release()

# رفع صورة
def upload_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # عرض الصورة الأصلية
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        
        # تحويل الصورة إلى مصفوفة NumPy
        image_array = np.array(image)
        
        # التأكد من الترتيب اللوني (PIL RGB -> OpenCV BGR) للمعالجة الصحيحة
        # لكن YOLO يقبل RGB، فقط الرسم يحتاج ضبط
        
        # تطبيق الكشف والرسم
        result_img = predict_and_draw(image_array)
        
        # النتيجة تكون NumPy Array، نحولها لصورة للعرض
        st.image(result_img, caption="Processed Image.", use_column_width=True)
        
        return Image.fromarray(result_img)
    return None

# تحميل الصورة المعدلة
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

# واجهة المستخدم Streamlit
st.title("Driver Distraction Detection (YOLOv8)")
st.write("Detect distracted driving behavior using YOLOv8.")

# خيارات للمستخدم
option = st.radio("Choose an option", ("Start Camera (Local Only)", "Upload Image"))

if option == "Start Camera (Local Only)":
    open_camera_and_detect()

if option == "Upload Image":
    image = upload_image()
    if image is not None:
        download_image(image)
