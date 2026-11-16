# Oxford Flowers 102 - Phân loại ảnh hoa bằng Machine Learning

Dự án phân loại ảnh thành **102 loài hoa** thuộc bộ dữ liệu Oxford Flowers 102. Phạm vi thực hiện sử dụng xử lý ảnh và đúng ba thuật toán học máy cổ điển: **KNN, Gaussian Naive Bayes và Decision Tree**. Dự án **không sử dụng Deep Learning**.

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
├── REPORT.md
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

`train_models.py` chạy cross-validation cho 21 tổ hợp bộ đặc trưng/thuật toán (7 bộ đặc trưng × 3 thuật toán). Huấn luyện có thể mất vài phút tùy cấu hình máy; terminal in tên mô hình và thời gian từng lượt CV.

Giải thích:
1. `eda.py`: kiểm tra số lượng ảnh, nhãn, split, ID trùng/lỗi và phân phối lớp.
2. `extract_features.py`: resize ảnh 128×128, trích xuất HOG, histogram HSV, LBP và SIFT Bag of Visual Words (BoVW); vocabulary BoVW được fit chỉ bằng descriptor của tập train. Lưu các tập `X_*.npy`, `y_*.npy` và tâm vocabulary.
3. `train_models.py`: so sánh KNN, Gaussian Naive Bayes và Decision Tree trên HOG, HSV, HOG+HSV, HOG+HSV+LBP, BoVW, HSV+BoVW và HOG+HSV+LBP+BoVW; chọn tham số bằng Stratified 3-fold CV trên train, chọn mô hình theo macro-F1 validation, fit lại trên train+validation và đánh giá test.
4. `app.py`: mở giao diện Gradio để tải ảnh và nhận diện.

## 5. Đặc trưng và mô hình

- **HOG**: ảnh xám, 9 hướng gradient, cell 8×8, block 2×2.
- **HSV**: histogram 16 bins cho H và 16 bins cho S, chuẩn hóa L1.
- **LBP**: histogram mã uniform với 8 điểm lân cận, bán kính 1; tính riêng trên từng ô của lưới 2×2.
- **SIFT BoVW**: SIFT tối đa 120 keypoint/ảnh, gom descriptor train thành vocabulary 64 từ; mỗi ảnh biểu diễn bằng histogram tần suất từ, chuẩn hóa L1.
- **Tập đặc trưng**: HOG (8.100 chiều), HSV (32), LBP (40), BoVW (64), HSV+BoVW (96), HOG+HSV+LBP (8.172), HOG+HSV+LBP+BoVW (8.236).
- **KNN**: thử số láng giềng và cách gán trọng số.
- **Gaussian Naive Bayes**: dùng `var_smoothing=1e-9` (đã là giá trị thắng CV trên 7 bộ đặc trưng).
- **Decision Tree**: so sánh tiêu chí chia nhánh, độ sâu và số mẫu lá tối thiểu; dùng `max_features="sqrt"` để giảm thời gian trên vector HOG lớn.
- **Tiền xử lý theo mô hình**: KNN dùng StandardScaler+PCA, ngoại trừ các tập có BoVW dùng chuẩn hóa vector và cosine distance. Gaussian Naive Bayes dùng StandardScaler, không PCA; Decision Tree dùng trực tiếp đặc trưng.
- **Chọn mô hình**: dùng macro-F1 trên validation sau khi tìm tham số bằng Stratified 3-fold CV trên train. Test được giữ lại cho đánh giá cuối.

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

Kết quả trong `outputs/experiments/` được sinh lại khi chạy pipeline. Báo cáo chỉ sử dụng số liệu được lưu từ lần chạy đó, không lấy kết quả cũ trong notebook làm kết quả chính thức.

## 7. Notebook

Các notebook `01_eda.ipynb` đến `04_error_analysis.ipynb` được giữ lại để tham khảo tiến trình làm việc ban đầu. Pipeline chính là các script trong `src/`; notebook cũ không phải nguồn kết quả chính thức.

## 8. Tài liệu

- Báo cáo kết quả thực nghiệm: [REPORT.md](REPORT.md)
- Khung báo cáo: [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md)
- Khai báo phạm vi AI hỗ trợ: [AI_USAGE.md](AI_USAGE.md)
- Chuẩn bị dữ liệu: [data/README.md](data/README.md)
