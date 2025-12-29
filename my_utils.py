import torch
import cv2
import numpy as np
from PIL import Image


# تحميل النموذج المدرب
def load_model(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    return model


# الكشف عن الأشياء في الصورة
def detect_objects(model, img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # إجراء الكشف باستخدام النموذج
    results = model(img_rgb)

    # عرض النتائج
    results.render()

    # تحويل الصورة إلى صورة لعرضها في Streamlit
    result_img = Image.fromarray(results.imgs[0])

    return result_img
