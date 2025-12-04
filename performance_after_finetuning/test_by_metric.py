
'''
======================KIỂM TRA HIỆU SUẤT SAU KHI FINETUNING BẰNG METRIC=======================
'''

from ultralytics import YOLO

PWD_dataset_folder = "../Dataset/Phat_project-3"
batch_size = 16
img_size = 640

model = YOLO(f"{PWD_dataset_folder}/best.pt")

# Đánh giá hiệu suất của mô hình trên tập validation.
# Sử dụng hàm model.val() với đường dẫn đến file cấu hình dữ liệu và các tham số batch size, kích thước ảnh.
metrics = model.val(data=f"{PWD_dataset_folder}/data.yaml", batch=batch_size, imgsz=img_size)
# In kết quả đánh giá.
print(metrics)