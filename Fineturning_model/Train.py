
'''=============================Train Model====================================='''

from ultralytics import YOLO

'''--- điền vào đường dẫn đến thư mục chứa dataset. ---'''
PWD_dataset_folder = ""  # Đường dẫn đến thư mục chứa dataset
epochs = 100  # Số epoch để huấn luyện
images_size = 640  # Kích thước ảnh
batch_size = 16  # Kích thước batch


model = YOLO("yolov8s.pt")  # Load the YOLOv8 model
model.train(data=f"../data.yaml", epochs=epochs, imgsz=images_size, batch=batch_size)