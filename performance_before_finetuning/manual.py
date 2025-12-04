
'''
======================KIỂM TRA HIỆU SUẤT TRƯỚC KHI FINETUNING THỦ CÔNG=======================
'''
import glob
import os
import random
from PIL import Image, ImageDraw
from ultralytics import YOLO
import matplotlib.pyplot as plt

PWD_dataset_folder = "../Dataset/Phat_project-3"
n_sample = 20

model = YOLO("yolov8s.pt")

images = glob.glob(f"{PWD_dataset_folder}/test/images/*.jpg")

for img_path in random.sample(images, n_sample):
    results = model.predict(img_path)

    # Lấy bounding box và id lớp từ kết quả dự đoán.
    boxes = results[0].boxes.xyxy
    class_ids = results[0].boxes.cls
    scores = results[0].boxes.conf

    img_pil = Image.open(img_path)
    draw = ImageDraw.Draw(img_pil)

    class_map = {0: "head", 1: "helmet", 2: "person"}
    color_map = {0: "red", 1: "blue", 2: "yellow"}

    # Lặp qua từng bounding box và id lớp.
    for box, cls in zip(boxes, class_ids):
        x_min, y_min, x_max, y_max = box.tolist()
        cls = int(cls)

        draw.rectangle([x_min, y_min, x_max, y_max], outline=color_map.get(cls,"white"), width=2)
        draw.text((x_min, max(y_min-12,0)), class_map.get(cls,"unknown"), fill=color_map.get(cls,"white"))

    plt.figure(figsize=(8,8))
    plt.imshow(img_pil)
    plt.axis('off')
    plt.show()