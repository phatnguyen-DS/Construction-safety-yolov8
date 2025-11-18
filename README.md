### **Hệ thống giám sát an toàn lao động trong công trường kết hợp YOLO V8 **


#### **Mục tiêu**: phát hiện và cảnh báo người không mang mũ bảo hộ vào cổng công trình, mục tiêu giảm sự phụ thuộc của con người khi rà soát người ra đảm bảo an toàn lao dộng, tiết kiệm nhân lực.
#### **Quy trình thực hiện dự án**:
##### Xác định vấn đề và bài toán: 
tình huống: mỗi khi có người ra vào công trình, cần có người rà soát xem người đó có đội mũ bảo hộ, đảm bảo an toàn lao dộng hay không, thay vào đó, hệ thống sử dụng camera vừa đảm bảo giảm nhân sự rà soát, đồng thời có thể thêm tính năng khi có người vi phạm lập tức phát loa cảnh báo, gởi hình ảnh cho ban quản lý an toàn lao động xử lý.
#####
##### phát triển model
###### - Thu thập dữ liệu: nguồn: Roboflow
###### - Anotation: ( vì bộ dữ liệu free nên chưa sạch và anotation chưa chuẩn nên cần kiểm tra và đánh nhãn lại)
###### - Thống kê và tìm hiểu dữ liệu: 
###### - Tăng cường dữ liệu:
###### - Training:
###### - Đánh giá mô hình:

##### Triển khai
###### - Phát triển kiến trúc hệ thống
###### - Xây dựng hệ thống
###### - Kiểm thử hệ thống

#### **Lưu ý**: vì anotation tương đối chuẩn và dể phân biệt nên quá trình huấn luyện chỉ thực hiện với 20 epochs

### Nếu bạn muốn train lại thì 
###### B1:down source code về, mở folder YOLO-V8, thêm vào file .env ROBoflow_API_KEY
###### B2: Cài thư viện cần thiết
'''
pip install -r requirement.txt
'''
###### B3: Tải bộ dữ liệu
'''
cd Dataset
python Get_dataset.py
'''
###### B4: Tăng cường dữ liệu
'''
cd ..
cd fix_imbalanced_data
python fix_imbalanced_data.py
'''
###### B5: Training(mặc định là 100 epochs)
'''
cd ..
cd Fineturning_model
python Train.py
'''
