
'''
======================KIỂM TRA HIỆU SUẤT TRƯỚC KHI FINETUNING BẰNG METRIC=======================
'''

from ultralytics import YOLO

PWD_dataset_folder = "../Dataset/Phat_project-3"
batch_size = 16
img_size = 640

model = YOLO("yolov8s.pt")

# Đánh giá hiệu suất của mô hình trên tập validation.
metrics = model.val(data=f"{PWD_dataset_folder}/data.yaml", batch=batch_size, imgsz=img_size)
# In kết quả đánh giá.
print(metrics)