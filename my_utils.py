# my_utils.py
from ultralytics import YOLO  # استيراد YOLO من مكتبة ultralytics
import cv2
import numpy as np
from PIL import Image

# تحميل النموذج المدرب باستخدام YOLOv8
def load_model(model_path):
    model = YOLO(model_path)  # استخدام YOLO من ultralytics
    return model

# الكشف عن الأشياء في الصورة
def detect_objects(model, img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # إجراء الكشف باستخدام النموذج
    results = model(img_rgb)

    # عرض النتائج
    results.render()  # رسم النتائج على الصورة

    # تحويل الصورة إلى صورة لعرضها في Streamlit
    result_img = Image.fromarray(results.imgs[0])

    return result_img
