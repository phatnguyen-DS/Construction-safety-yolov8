''''
============ XUẤT DỮ LIỆU KHÔNG CÂN BẰNG ============

'''
import pandas as pd
import os
import yaml
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random   
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


PWD_dataset_folder = "../Dataset/Phat_project-3"

# Đọc file CSV đã lưu trước đó.
table = pd.read_csv("table.csv", index_col=0)

# Xác định số lượng ảnh hiện có cho mỗi lớp ('head', 'helmet') từ bảng thống kê table.
# Bảng DataFrame 'table' đã chứa thông tin này cho tập 'train', được sử dụng để huấn luyện.
head_count = table.loc['0', 'train']
helmet_count = table.loc['1', 'train']

# Xác định số lượng ảnh tối đa trong các lớp, đây sẽ là mục tiêu cân bằng.
target_count = helmet_count

# Tính toán số lượng ảnh cần tăng cường cho mỗi lớp ('head', 'person') bằng cách lấy số lượng ảnh mục tiêu trừ đi số lượng ảnh hiện có của lớp đó.
needed_head = target_count - head_count
print(f"Số lượng ảnh hiện có của lớp 'head': {head_count}")
print(f"Số lượng ảnh hiện có của lớp 'helmet': {helmet_count}")
print(f"Số lượng ảnh mục tiêu (của lớp 'helmet'): {target_count}")
print(f"Số lượng ảnh cần tăng cường cho lớp 'head': {needed_head}")

'''==================================='''
#Tạo một từ điển để lưu trữ đường dẫn ảnh và nhãn cho từng lớp.
class_paths = {
    0: {"images": [], "labels": []},  # head
    1: {"images": [], "labels": []}   # helmet
}

#Lặp qua tất cả các tệp nhãn trong thư mục huấn luyện.
label_dir = f"{PWD_dataset_folder}/train/labels"
image_dir = f"{PWD_dataset_folder}/train/images"

for label_filename in os.listdir(label_dir):
    if label_filename.endswith(".txt"):
        label_path = os.path.join(label_dir, label_filename)
        image_filename = label_filename.replace(".txt", ".jpg")
        image_path = os.path.join(image_dir, image_filename)

        classes_in_image = set()
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        classes_in_image.add(class_id)

        for class_id in class_paths.keys():
            if class_id in classes_in_image:
                class_paths[class_id]["images"].append(image_path)
                class_paths[class_id]["labels"].append(label_path)

# In số lượng ảnh tìm thấy cho mỗi lớp (nên tương ứng với số lượng tệp nhãn chứa lớp đó)
print(f"Số lượng ảnh tìm thấy cho lớp 'head' (0): {len(class_paths[0]['images'])}")
print(f"Số lượng ảnh tìm thấy cho lớp 'helmet' (1): {len(class_paths[1]['images'])}")


def augment_image_and_labels(image_path, label_path, num_augmentations, output_image_dir, output_label_dir, augmenter):
    """
    Áp dụng các phép tăng cường ngẫu nhiên cho ảnh và các nhãn YOLO tương ứng sử dụng Albumentations.

    Args:
        image_path (str): Đường dẫn đến ảnh đầu vào.
        label_path (str): Đường dẫn đến tệp nhãn đầu vào (định dạng YOLO).
        num_augmentations (int): Số lượng phiên bản tăng cường cần tạo.
        output_image_dir (str): Thư mục để lưu ảnh đã tăng cường.
        output_label_dir (str): Thư mục để lưu nhãn đã tăng cường.
        augmenter (A.Compose): Đối tượng Albumentations Compose để tăng cường.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Không thể đọc ảnh {image_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Chuyển đổi sang RGB cho Albumentations

        # Đọc nhãn
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        labels.append([class_id, x_center, y_center, width, height])

        # Albumentations mong đợi bounding box ở định dạng [x_min, y_min, x_max, y_max] hoặc [class_id, x_c, y_c, w, h] đã chuẩn hóa
        # Chúng ta sẽ sử dụng định dạng [x_c, y_c, w, h] đã chuẩn hóa và chỉ định định dạng.
        bboxes = [[label[1], label[2], label[3], label[4], int(label[0])] for label in labels] # Định dạng: [x_c, y_c, w, h, class_id]

        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        for i in range(num_augmentations):

            # Áp dụng tăng cường sử dụng Albumentations
            augmented = augmenter(image=img, bboxes=bboxes)
            augmented_img = augmented['image']
            augmented_bboxes = augmented['bboxes']

            # Chuyển đổi ảnh đã tăng cường trở lại BGR để lưu bằng cv2
            augmented_img_bgr = cv2.cvtColor(np.array(augmented_img), cv2.COLOR_RGB2BGR)

            # Lưu ảnh và nhãn đã tăng cường
            augmented_image_filename = f"{base_filename}_aug_{i}.jpg"
            augmented_label_filename = f"{base_filename}_aug_{i}.txt"
            augmented_image_path = os.path.join(output_image_dir, augmented_image_filename)
            augmented_label_path = os.path.join(output_label_dir, augmented_label_filename)

            cv2.imwrite(augmented_image_path, augmented_img_bgr)

            with open(augmented_label_path, 'w') as f:
                for bbox in augmented_bboxes:
                    # Albumentations trả về [x_c, y_c, w, h, class_id] nếu chúng ta chỉ định định dạng 'yolo'
                    # Chúng ta cần ghi lại nó ở định dạng YOLO: class_id x_c y_c w h
                    class_id = int(bbox[4])
                    x_c, y_c, w, h = bbox[:4]
                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    except Exception as e:
        print(f"Lỗi khi tăng cường ảnh {image_path}: {e}")


# Định nghĩa thư mục đầu ra
augmented_train_image_dir = f"{PWD_dataset_folder}/train/augmented_images"
augmented_train_label_dir = f"{PWD_dataset_folder}/train/augmented_labels"

# Tạo thư mục đầu ra nếu chúng chưa tồn tại
os.makedirs(augmented_train_image_dir, exist_ok=True)
os.makedirs(augmented_train_label_dir, exist_ok=True)

# Định nghĩa bộ tăng cường Albumentations
augment = A.Compose([
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.HueSaturationValue(p=0.3),
], bbox_params=A.BboxParams(format='yolo', clip=True))

# Tính toán số lượng tăng cường cần thiết cho mỗi ảnh của mỗi lớp
num_head_images = len(class_paths[0]['images'])

# Phân phối số lượng tăng cường cho các ảnh hiện có
if num_head_images > 0:
    augmentations_per_head_image = needed_head // num_head_images
else:
    augmentations_per_head_image = 0

print(f"Tăng cường lớp 'head'. Cần: {needed_head}, Ảnh: {num_head_images}, Tăng cường mỗi ảnh: {augmentations_per_head_image}")

augmented_head_count = 0

# Áp dụng tăng cường cho lớp 'head'
if num_head_images > 0:
    for img_path, label_path in zip(class_paths[0]['images'], class_paths[0]['labels']):
        current_augmentations = augmentations_per_head_image  # Không thêm số dư
        augment_image_and_labels(
            img_path, label_path,
            current_augmentations,
            augmented_train_image_dir,
            augmented_train_label_dir,
            augment
        )
        augmented_head_count += current_augmentations

print(f"\nTổng số ảnh tăng cường đã tạo cho lớp 'head': {augmented_head_count}")
print(f"Tổng số ảnh tăng cường đã tạo: {augmented_head_count}")

'''========================================================================='''

# Xác định đường dẫn đến file data.yaml
data_yaml_path = f"{PWD_dataset_folder}/data.yaml"

# Xác định đường dẫn đến các thư mục đã tăng cường
augmented_train_image_dir = f"{PWD_dataset_folder}/train/augmented_images"

# Tải file data.yaml hiện có
with open(data_yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)

# Cập nhật đường dẫn 'train' để chỉ bao gồm các thư mục ảnh gốc và đã tăng cường
# Ensure the original path is correct and add the augmented path
original_train_image_dir = f"{PWD_dataset_folder}/train/images" # Corrected original path

data_yaml['train'] = [original_train_image_dir, augmented_train_image_dir]

# Lưu lại file data.yaml đã cập nhật
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_yaml, f)

print(f"Đã cập nhật {data_yaml_path} để bao gồm đường dẫn dữ liệu đã tăng cường.")
print("Đường dẫn 'train' mới trong data.yaml:")
print(data_yaml['train'])

'''========================================================================='''

'''========================= kiem tra du lieu sau can bang ======================='''

# Xác định đường dẫn đến các thư mục nhãn gốc và đã tăng cường cho tập huấn luyện.
original_train_label_dir = f"{PWD_dataset_folder}/train/labels"
augmented_train_label_dir = f"{PWD_dataset_folder}/train/augmented_labels"

# Đường dẫn đến các thư mục nhãn validation và test
val_label_dir = f"{PWD_dataset_folder}/valid/labels"
test_label_dir = f"{PWD_dataset_folder}/test/labels"


# Khởi tạo một từ điển dict_ để đếm số lần xuất hiện của mỗi lớp (0, 1) trong các tập 'train', 'val', và 'test'.
dict_ = {
         "train":{
             "0":0,
             "1":0
         },
         "val": {
             "0":0,
             "1":0
         },
         "test": {
             "0":0,
             "1":0
         }
         }

# Định nghĩa hàm count nhận đường dẫn thư mục và tên tập dữ liệu ('train', 'val', hoặc 'test') làm đầu vào.
def count(folder_path: str, tap: str):
  """
  Đếm số lần xuất hiện của mỗi lớp trong các tệp nhãn bên trong một thư mục được chỉ định.

  Args:
      folder_path (str): Đường dẫn đến thư mục chứa các tệp nhãn.
      tap (str): Tên của phần dữ liệu ('train', 'val', hoặc 'test').
  """
  # Kiểm tra xem thư mục có tồn tại không.
  if not os.path.exists(folder_path):
      print(f"Cảnh báo: Không tìm thấy thư mục: {folder_path}")
      return

  # Liệt kê tất cả các tệp trong thư mục.
  ls_dir = os.listdir(folder_path)

  # Tạo danh sách đường dẫn đầy đủ cho các tệp .txt trong thư mục.
  ls_path = [os.path.join(folder_path, file) for file in ls_dir if file.endswith('.txt')] # Chỉ xử lý các tệp .txt
  
  # Lặp qua từng đường dẫn tệp nhãn.
  for path in ls_path:
    try:

        # Mở tệp và đọc từng dòng.
        with open(path, "r") as f:
          for line in f:

              # Tách các phần của dòng.
              parts = line.split()

              # Đảm bảo dòng không trống.
              if len(parts) > 0:
                  class_id = int(parts[0])
                  if str(class_id) in dict_[f'{tap}']:
                      dict_[f'{tap}'][f'{class_id}'] +=1

    # Bắt ngoại lệ nếu có lỗi khi đọc tệp nhãn.
    except Exception as e:
        print(f"Lỗi khi đọc tệp nhãn {path}: {e}")

# Gọi hàm count cho thư mục nhãn huấn luyện gốc.
count(original_train_label_dir, "train")

# Gọi hàm count cho thư mục nhãn huấn luyện đã tăng cường.
count(augmented_train_label_dir, "train")

# Gọi hàm count cho thư mục nhãn validation và test.
count(val_label_dir, "val")
count(test_label_dir, "test")

# Tạo một bảng pandas DataFrame từ dict_ để hiển thị số lượng lớp cho mỗi tập dữ liệu.
table = pd.DataFrame(dict_)

# In bảng DataFrame.
print("Phân bố lớp sau khi tăng cường dữ liệu:")
display(table)

# Tính phần trăm phân bố lớp trong tập 'train'.
percent_train = table['train'].div(table['train'].sum()) * 100

# Tạo biểu đồ tròn để trực quan hóa phần trăm phân bố lớp trong tập 'train'.
labels = ['head', 'helmet'] # Giả sử class_id 0:head, 1:helmet, 2:person
colors = sns.color_palette('pastel')[0:len(percent_train)]

plt.figure(figsize=(6,6))
plt.pie(percent_train, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Phân bố lớp sau khi tăng cường dữ liệu (Tập Train)')
plt.show()