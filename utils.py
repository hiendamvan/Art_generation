# utils.py
# Các hàm hỗ trợ (load ảnh, hiển thị ảnh, lưu kết quả,...)

import tensorflow as tf
import numpy as np
import cv2

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    image = cv2.resize(image, (400, 400))
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)  # Thêm chiều batch

def save_image(image, path):
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))