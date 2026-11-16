# Báo cáo: Phân loại 102 loài hoa bằng học máy cổ điển

## 1. Mục tiêu và phạm vi

Xây dựng pipeline phân loại ảnh Oxford Flowers 102 thành 102 lớp bằng xử lý ảnh và ba thuật toán: **KNN, Gaussian Naive Bayes và Decision Tree**. Dự án không sử dụng Deep Learning.

## 2. Dữ liệu

Bộ dữ liệu có 8.189 ảnh. Thực nghiệm sử dụng split chính thức:

| Tập | Số ảnh |
|---|---:|
| Train | 1.020 |
| Validation | 1.020 |
| Test | 6.149 |

Nhãn và ID split được đọc từ `imagelabels.mat` và `setid.mat`. Script EDA kiểm tra ID trùng, ảnh thiếu và phân phối lớp trước khi chạy pipeline.

## 3. Tiền xử lý và đặc trưng

Ảnh được resize về 128 × 128 pixel. Mỗi ảnh được biểu diễn bằng:

- **HOG** trên ảnh xám: 9 hướng gradient, cell 8 × 8, block 2 × 2; 8.100 chiều.
- **HSV**: histogram 16 bins cho kênh H và 16 bins cho kênh S, chuẩn hóa L1; 32 chiều.
- **HOG+HSV**: nối hai vector trên; 8.132 chiều.

## 4. Mô hình và thiết kế thực nghiệm

Ba thuật toán được so sánh trên cả ba bộ đặc trưng:

- **KNN**: `n_neighbors` = 3, 5, 7; `weights` = uniform hoặc distance.
- **Gaussian Naive Bayes**: `var_smoothing` = 1e-9, 1e-8, 1e-7.
- **Decision Tree**: criterion = gini hoặc entropy; `max_depth` = None hoặc 30; `min_samples_leaf` = 1 hoặc 2.

KNN và Naive Bayes dùng `StandardScaler` và PCA trong pipeline; Decision Tree dùng trực tiếp đặc trưng. Số chiều PCA được tính riêng cho từng tập đặc trưng: `min(200, số_mẫu_train - 1, số_đặc_trưng)`. Do đó HOG dùng tối đa 200 thành phần, còn HSV dùng 32.

Tham số được chọn bằng Stratified 3-fold CV chỉ trên train với macro-F1. Mô hình cuối được chọn theo macro-F1 validation, fit lại trên train + validation, sau đó đánh giá một lần trên test.

## 5. Kết quả validation

Macro-F1 validation là tiêu chí lựa chọn mô hình:

| Đặc trưng | Thuật toán | Accuracy | Macro-F1 |
|---|---|---:|---:|
| HOG | KNN | 0.0422 | 0.0274 |
| HOG | Naive Bayes | 0.0627 | 0.0483 |
| HOG | Decision Tree | 0.0225 | 0.0224 |
| HSV | KNN | 0.1902 | **0.1762** |
| HSV | Naive Bayes | 0.1716 | 0.1666 |
| HSV | Decision Tree | 0.1304 | 0.1261 |
| HOG+HSV | KNN | 0.0431 | 0.0276 |
| HOG+HSV | Naive Bayes | 0.0686 | 0.0547 |
| HOG+HSV | Decision Tree | 0.0618 | 0.0526 |

Mô hình được chọn là **HSV + KNN**, với `n_neighbors=3` và `weights=distance`.

## 6. Kết quả test của mô hình được chọn

| Chỉ số | Giá trị |
|---|---:|
| Accuracy | 0.2083 |
| Precision macro | 0.1876 |
| Recall macro | 0.2167 |
| F1 macro | 0.1887 |
| F1 weighted | 0.2045 |
| Top-5 accuracy | 0.3220 |

Các giá trị này được đọc từ `outputs/experiments/final_test_metrics.json`. Bảng validation đầy đủ nằm ở `outputs/experiments/validation_comparison.csv`.

## 7. Phân tích lỗi

Theo `classification_report.csv`, các lớp có recall thấp nhất trong lần chạy này gồm mallow (0.000), hibiscus (0.018), sword lily (0.018), cyclamen (0.022) và sweet pea (0.028).

Một số nhầm lẫn có số lượng cao trong confusion matrix là passion flower → toad lily (16 ảnh), passion flower → bird of paradise (16), thorn apple → tree poppy (16), watercress → english marigold (15), và passion flower → wild pansy (14). Đây là thống kê của bộ test, không phải đánh giá nguyên nhân thị giác cho từng ảnh.

## 8. Hạn chế và kết luận

Macro-F1 test 0.1887 cho thấy mô hình còn khó phân biệt 102 lớp. HSV + KNN là cấu hình tốt nhất trong các cấu hình đã thử, nhưng histogram màu bỏ qua bố cục và hình dạng; HOG hoặc đặc trưng kết hợp không cải thiện kết quả trong thực nghiệm này. Kết quả không nên được diễn giải như một hệ thống nhận diện có độ chính xác cao.

## 9. Tái lập

Từ thư mục gốc, cài thư viện bằng `pip install -r requirements.txt`, chuẩn bị dữ liệu theo [data/README.md](data/README.md), rồi chạy:

```bash
python src/eda.py
python src/extract_features.py
python src/train_models.py
python src/app.py
```

Kết quả nêu trong báo cáo lấy từ các tệp đầu ra của pipeline trong workspace. Khi chạy lại với dữ liệu hoặc phiên bản thư viện khác, hãy thay số liệu bằng kết quả mới được sinh ra.
