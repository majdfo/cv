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
    model = YOLO('best.pt')
    return model

model = load_model()

# دالة لتحديد سبب تشدد السائق ورسم المستطيلات
def predict_and_draw(frame):
    results = model.predict(frame)  # الحصول على نتائج من النموذج
    img = frame.copy()
    
    # رسم مستطيلات حول الكائنات المكتشفة
    for pred in results.pred[0]:
        if pred.conf[0] > 0.25:  # استخدم قيمة الثقة المناسبة
            x1, y1, x2, y2 = map(int, pred.xyxy[0])
            label = model.names[int(pred.cls[0])]
            color = (0, 255, 0) if label == "PhoneUse" else (255, 0, 0) if label == "Seatbelt" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return img

# دالة لفتح الكاميرا ورسم المستطيلات في الوقت الحقيقي
def open_camera_and_detect():
    st.text("Opening camera...")
    cap = cv2.VideoCapture(0)  # فتح كاميرا الويب (0: الكاميرا الافتراضية)

    if not cap.isOpened():
        st.error("Unable to access camera")
        return

    while True:
        ret, frame = cap.read()  # قراءة الإطار من الكاميرا
        if not ret:
            st.error("Failed to grab frame")
            break

        # كشف الكائنات ورسم المستطيلات
        frame_with_boxes = predict_and_draw(frame)
        
        # تحويل الصورة من BGR إلى RGB
        frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        
        # عرض الصورة في Streamlit
        st.image(frame_rgb, channels="RGB", use_column_width=True)

        # التحكم في الوقت بين الإطارات لتجنب التحميل الزائد
        time.sleep(0.1)  # تأخير صغير لتمكين التفاعل

        # كود لإيقاف الكاميرا بعد فترة أو عند النقر على زر الإيقاف
        if st.button('Stop Camera'):
            cap.release()
            st.success("Camera stopped.")
            break

# رفع صورة
def upload_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        
        # تحويل الصورة إلى مصفوفة NumPy ليتم معالجتها
        image_array = np.array(image)
        
        # تطبيق الكشف والرسم
        result_img = predict_and_draw(image_array)
        
        # تحويل الصورة المعدلة من NumPy إلى Image
        result_image = Image.fromarray(result_img)
        
        # عرض الصورة المعدلة
        st.image(result_image, caption="Processed Image with Bounding Boxes.", use_column_width=True)
        
        # العودة بالصورة المعدلة
        return result_image
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
st.title("Driver Distraction Detection")
st.subheader("Open camera to detect distracted driving behavior in real-time or upload an image to process")

# خيارات للمستخدم: بدء الكاميرا أو رفع صورة
option = st.radio("Choose an option", ("Start Camera", "Upload Image"))

if option == "Start Camera":
    open_camera_and_detect()

if option == "Upload Image":
    image = upload_image()
    if image is not None:
        download_image(image)
