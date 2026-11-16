# Oxford Flowers 102 - Phân loại ảnh hoa bằng Machine Learning

Dự án phân loại ảnh thành **102 loài hoa** thuộc bộ dữ liệu Oxford Flowers 102. Phạm vi thực hiện sử dụng xử lý ảnh và các thuật toán học máy cổ điển (HOG, HSV, KNN, SVM), **không sử dụng Deep Learning**.

## 1. Cấu trúc dự án

```text
MachineLearning/
├── data/
│   ├── raw/                 # Dữ liệu gốc (tự tải, không commit)
│   ├── processed/           # Đặc trưng .npy (tự sinh, không commit)
│   └── README.md
├── models/                  # Mô hình .joblib (tự sinh, không commit)
├── outputs/
│   ├── eda/
│   └── experiments/
├── src/
│   ├── eda.py
│   ├── extract_features.py
│   ├── train_models.py
│   ├── save_models.py       # Script lưu mô hình KNN phiên bản cũ
│   └── app.py
├── 01_eda.ipynb
├── 02_feature_extraction.ipynb
├── 03_model_training.ipynb
├── 04_error_analysis.ipynb
├── REPORT_TEMPLATE.md
├── AI_USAGE.md
└── requirements.txt
```

## 2. Cài đặt môi trường

Khuyến nghị Python 3.10 hoặc 3.11.

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường, có thể dùng Command Prompt:

```bat
.venv\Scripts\activate.bat
```

## 3. Tải dữ liệu

Tải Oxford Flowers 102 từ trang chính thức Oxford VGG:

- Ảnh: `102flowers.tgz`
- Nhãn: `imagelabels.mat`
- Chia tập: `setid.mat`

Đặt dữ liệu theo cấu trúc:

```text
data/raw/
├── jpg/
│   ├── image_00001.jpg
│   ├── image_00002.jpg
│   └── ...
├── imagelabels.mat
└── setid.mat
```

Không đưa dữ liệu gốc, đặc trưng đã trích xuất hoặc mô hình nhị phân lên Git vì dung lượng lớn. Hướng dẫn chi tiết: [data/README.md](data/README.md).

## 4. Chạy toàn bộ pipeline

Chạy các lệnh dưới đây từ thư mục gốc của repository:

```bash
python src/eda.py
python src/extract_features.py
python src/train_models.py
python src/app.py
```

Giải thích:
1. `eda.py`: kiểm tra số lượng ảnh, nhãn, split, ID trùng/lỗi và phân phối lớp.
2. `extract_features.py`: resize ảnh 128×128, trích xuất HOG + histogram HSV và lưu các tập `X_*.npy`, `y_*.npy`.
3. `train_models.py`: so sánh KNN baseline và SVM trên HOG, HSV, HOG+HSV; lựa chọn theo macro-F1 validation; fit lại trên train+validation; đánh giá test.
4. `app.py`: mở giao diện Gradio để tải ảnh và nhận diện.

## 5. Đặc trưng và mô hình

- **HOG**: ảnh xám, 9 hướng gradient, cell 8×8, block 2×2.
- **HSV**: histogram 16 bins cho H và 16 bins cho S, chuẩn hóa L1.
- **KNN**: mô hình baseline để đối chiếu.
- **SVM**: thử kernel linear/RBF và các giá trị C; lựa chọn tham số bằng Stratified 3-fold CV trên train.
- **Scaling và PCA**: nằm trong pipeline để chỉ fit trên dữ liệu huấn luyện của từng fold, tránh rò rỉ dữ liệu.
- **Chọn mô hình**: dùng macro-F1 trên validation. Test được giữ lại cho đánh giá cuối.

Lưu ý: mỗi lớp có ít ảnh trong tập train chính thức. Đừng kỳ vọng đặc trưng thủ công sẽ đạt độ chính xác như mô hình học sâu.

## 6. Đầu ra

Sau khi chạy, kiểm tra:
- `outputs/eda/dataset_summary.json`
- `outputs/eda/class_distribution.png`
- `outputs/experiments/validation_comparison.csv`
- `outputs/experiments/final_test_metrics.json`
- `outputs/experiments/classification_report.csv`
- `outputs/experiments/confusion_matrix.png`
- `models/best_model.joblib`

**Không điền số liệu kết quả vào báo cáo trước khi chạy pipeline.** Dùng `REPORT_TEMPLATE.md` làm khung báo cáo và điền số liệu thực tế từ các file đầu ra.

## 7. Notebook

Các notebook `01_eda.ipynb` đến `04_error_analysis.ipynb` được giữ lại để tham khảo tiến trình làm việc ban đầu. Pipeline mới được tổ chức thành script để chạy lại nhất quán. Khi trình bày, cần phân biệt rõ kết quả notebook cũ và kết quả mới.

## 8. Tài liệu

- Khung báo cáo: [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md)
- Khai báo phạm vi AI hỗ trợ: [AI_USAGE.md](AI_USAGE.md)
- Chuẩn bị dữ liệu: [data/README.md](data/README.md)
