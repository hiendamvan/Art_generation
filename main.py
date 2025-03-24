# main.py
# Chương trình chính để chạy Neural Style Transfer

from style_transfer import NeuralStyleTransfer
from utils import load_image, save_image
import config

def main():
    # Load ảnh nội dung và phong cách
    content_image = load_image(config.CONTENT_IMAGE_PATH)
    style_image = load_image(config.STYLE_IMAGE_PATH)
    
        
    print("Content image shape:", content_image.shape)  # Phải là (1, h, w, 3)
    print("Style image shape:", style_image.shape)
    # Khởi tạo và thực hiện style transfer
    nst = NeuralStyleTransfer(content_image, style_image)
    output_image = nst.train_style_transfer()
    
    # Lưu ảnh kết quả
    save_image(output_image, config.OUTPUT_IMAGE_PATH)
    print("Quá trình hoàn thành! Ảnh đã lưu tại:", config.OUTPUT_IMAGE_PATH)

if __name__ == "__main__":
    main()