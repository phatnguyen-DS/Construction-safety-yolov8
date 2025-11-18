
'''
======================KIỂM TRA HIỆU SUẤT SAU KHI FINETUNING BẰNG METRIC=======================
'''

from ultralytics import YOLO

# điền vào đường dẫn đến thư mục chứa dataset.
PWD_dataset_folder = ""  # Đường dẫn đến thư mục chứa dataset
batch_size = 16  # Kích thước batch
img_size = 640  # Kích thước ảnh

# Đường dẫn đến file cấu hình dữ liệu.

model = YOLO(f"{PWD_dataset_folder}/best.pt")  # Load the YOLOv8 model

# Đánh giá hiệu suất của mô hình trên tập validation.
# Sử dụng hàm model.val() với đường dẫn đến file cấu hình dữ liệu và các tham số batch size, kích thước ảnh.
metrics = model.val(data=f"{PWD_dataset_folder}/data.yaml", batch=batch_size, imgsz=img_size)
# In kết quả đánh giá.
print(metrics)