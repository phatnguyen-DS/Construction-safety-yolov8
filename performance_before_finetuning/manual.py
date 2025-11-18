
'''
======================KIỂM TRA HIỆU SUẤT TRƯỚC KHI FINETUNING THỦ CÔNG=======================
'''
import glob
import os
import random
from PIL import Image, ImageDraw
from ultralytics import YOLO
import matplotlib.pyplot as plt

'''--- điền vào đường dẫn đến thư mục chứa dataset. ---'''
PWD_dataset_folder = "../Dataset/Phat_project-3"  # Đường dẫn đến thư mục chứa dataset
n_sample = 20  # Số lượng ảnh mẫu để kiểm tra

model = YOLO("yolov8s.pt")  # Load the YOLOv8 model

# Lấy danh sách tất cả các tệp ảnh .jpg trong thư mục thử nghiệm.
images = glob.glob(f"{PWD_dataset_folder}/test/images/*.jpg")
# Chọn ngẫu nhiên 20 ảnh từ danh sách để dự đoán.
for img_path in random.sample(images, n_sample):
    # Sử dụng mô hình YOLO để dự đoán trên ảnh.
    results = model.predict(img_path)

    # Lấy bounding box và id lớp từ kết quả dự đoán.
    boxes = results[0].boxes.xyxy
    class_ids = results[0].boxes.cls  # class id
    scores = results[0].boxes.conf
    # Mở ảnh bằng Pillow.
    img_pil = Image.open(img_path)
    # Tạo đối tượng ImageDraw để vẽ lên ảnh.
    draw = ImageDraw.Draw(img_pil)

    # Ánh xạ id lớp → tên lớp và màu sắc.
    class_map = {0: "head", 1: "helmet", 2: "person"}
    color_map = {0: "red", 1: "blue", 2: "yellow"}

    # Lặp qua từng bounding box và id lớp.
    for box, cls in zip(boxes, class_ids):
        # Lấy tọa độ bounding box.
        x_min, y_min, x_max, y_max = box.tolist()
        # Chuyển id lớp sang kiểu số nguyên.
        cls = int(cls)
        # Vẽ hình chữ nhật (bounding box) lên ảnh với màu tương ứng.
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color_map.get(cls,"white"), width=2)
        # Vẽ tên lớp lên bounding box.
        draw.text((x_min, max(y_min-12,0)), class_map.get(cls,"unknown"), fill=color_map.get(cls,"white"))

    # Hiển thị ảnh bằng matplotlib.
    plt.figure(figsize=(8,8))
    plt.imshow(img_pil)
    # Tắt trục tọa độ.
    plt.axis('off')
    # Hiển thị biểu đồ.
    plt.show()