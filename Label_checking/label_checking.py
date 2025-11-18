'''
   ========================== KIỂM TRA NHÃN ==========================
   Mục đích: Kiểm tra nhãn của các ảnh trong tập huấn luyện
'''
import glob
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random

# Đường dẫn đến thư mục chứa dataset.
PWD_dataset_folder = ""  # Dien vao duong dan den folder chua dataset

# Số lượng ảnh mẫu để kiểm tra.
n_sample = 20
# Ánh xạ id lớp → tên lớp và màu sắc
class_map = {
    0: ("head", "red"),
    1: ("helmet", "blue")
}

# Lấy danh sách tất cả các tệp ảnh .jpg trong thư mục huấn luyện.
images = glob.glob(f"{PWD_dataset_folder}/train/images/*.jpg")
# Chọn ngẫu nhiên 20 ảnh từ danh sách để kiểm tra.
for img_path in random.sample(images, n_sample):
    # Lấy tên file nhưng bỏ phần mở rộng.
    filename_wo_ext = os.path.splitext(os.path.basename(img_path))[0]

    # Tạo đường dẫn file label tương ứng.
    label_path = os.path.join(f"{PWD_dataset_folder}/train/labels", filename_wo_ext + ".txt")

    # Kiểm tra xem file label có tồn tại không.
    if os.path.exists(label_path):
        # Đọc nội dung label.
        with open(label_path, 'r') as f:
            content = f.read().strip()

        # Mở ảnh bằng Pillow.
        img_ = Image.open(img_path)
        # Tạo đối tượng ImageDraw để vẽ lên ảnh.
        draw = ImageDraw.Draw(img_)
        # Lấy kích thước ảnh (chiều rộng W, chiều cao H).
        W, H = img_.size

        # Nếu file label không rỗng.
        if content:
            # Lặp qua từng dòng trong nội dung label.
            for line in content.splitlines():
                # Tách các giá trị trong dòng.
                parts = line.split()
                # Bỏ qua dòng không đúng định dạng (không có 5 phần).
                if len(parts) != 5:
                    continue

                # Chuyển đổi các giá trị sang kiểu float.
                class_id, x_center, y_center, w, h = map(float, parts)

                # Chuyển tọa độ chuẩn hóa sang tọa độ pixel.
                x_center *= W
                y_center *= H
                w *= W
                h *= H

                # Tính toán tọa độ góc trên bên trái và góc dưới bên phải của bounding box.
                x_min = x_center - w/2
                y_min = y_center - h/2
                x_max = x_center + w/2
                y_max = y_center + h/2

                # Lấy tên class và màu sắc dựa vào class_id.
                class_name, color = class_map.get(int(class_id), ("unknown", "white"))

                # Vẽ hình chữ nhật (bounding box) lên ảnh.
                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)

                # Vẽ tên class trên box.
                draw.text((x_min, max(y_min-12, 0)), class_name, fill=color)

        # Hiển thị ảnh bằng matplotlib.
        plt.figure(figsize=(8,8))
        plt.imshow(img_)
        # Tắt trục tọa độ.
        plt.axis('off')
        # Hiển thị biểu đồ.
        plt.show()