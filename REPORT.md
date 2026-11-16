# Báo cáo: Phân loại 102 loài hoa bằng BoVW và học máy cổ điển

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

Ảnh được resize về 128 × 128 pixel. Các đặc trưng được trích xuất:

- **HOG** trên ảnh xám: 9 hướng gradient, cell 8 × 8, block 2 × 2; 8.100 chiều.
- **HSV**: histogram 16 bins cho kênh H và 16 bins cho kênh S, chuẩn hóa L1; 32 chiều.
- **LBP**: histogram uniform với 8 điểm lân cận, bán kính 1; histogram trên lưới 2 × 2; 40 chiều.
- **SIFT BoVW**: tối đa 120 keypoint/ảnh, descriptor được gom thành vocabulary 64 từ bằng MiniBatchKMeans. Vocabulary chỉ fit từ descriptor tập train; mỗi ảnh được biểu diễn bằng histogram 64 chiều chuẩn hóa L1.
- So sánh HOG, HSV, HOG+HSV, HOG+HSV+LBP, BoVW, HSV+BoVW và HOG+HSV+LBP+BoVW; vector đầy đủ có 8.236 chiều.

## 4. Mô hình và thiết kế thực nghiệm

Ba thuật toán cổ điển được so sánh trên bảy bộ đặc trưng:

- **KNN**: `n_neighbors` = 3, 5, 7; `weights` = uniform hoặc distance.
- **Gaussian Naive Bayes**: dùng `var_smoothing=1e-9`, giá trị thắng CV ở tất cả các bộ đặc trưng trong lượt thử tham số ban đầu.
- **Decision Tree**: criterion = gini hoặc entropy; `max_depth` = None hoặc 30; `min_samples_leaf` = 1 hoặc 2; `max_features` = sqrt để hạn chế số đặc trưng xét tại mỗi nút.

KNN dùng `StandardScaler` và PCA, ngoại trừ các bộ đặc trưng có BoVW dùng chuẩn hóa vector và cosine distance. Gaussian Naive Bayes dùng `StandardScaler` không kèm PCA; Decision Tree dùng trực tiếp đặc trưng. PCA được giới hạn theo `min(200, số_mẫu_train - 1, số_đặc_trưng)` và fit bên trong từng fold.

Tham số được chọn bằng Stratified 3-fold CV chỉ trên train với macro-F1. Mô hình cuối được chọn theo macro-F1 validation, fit lại trên train + validation, sau đó đánh giá trên test.

## 5. Kết quả validation

Macro-F1 validation là tiêu chí lựa chọn mô hình:

| Đặc trưng | Thuật toán | Accuracy | Macro-F1 |
|---|---|---:|---:|
| HOG | KNN | 0.0422 | 0.0274 |
| HOG | Naive Bayes | 0.1098 | 0.1195 |
| HOG | Decision Tree | 0.0196 | 0.0193 |
| HSV | KNN | 0.1902 | 0.1762 |
| HSV | Naive Bayes | 0.1686 | 0.1547 |
| HSV | Decision Tree | 0.1176 | 0.1146 |
| HOG+HSV | KNN | 0.0431 | 0.0276 |
| HOG+HSV | Naive Bayes | 0.1461 | 0.1619 |
| HOG+HSV | Decision Tree | 0.0294 | 0.0271 |
| HOG+HSV+LBP | KNN | 0.0422 | 0.0248 |
| HOG+HSV+LBP | Naive Bayes | 0.1510 | 0.1697 |
| HOG+HSV+LBP | Decision Tree | 0.0353 | 0.0335 |
| BoVW | KNN | 0.0775 | 0.0709 |
| BoVW | Naive Bayes | 0.0873 | 0.0864 |
| BoVW | Decision Tree | 0.0225 | 0.0216 |
| HSV+BoVW | KNN | 0.2265 | 0.2111 |
| HSV+BoVW | Naive Bayes | 0.2353 | **0.2371** |
| HSV+BoVW | Decision Tree | 0.0706 | 0.0668 |
| HOG+HSV+LBP+BoVW | KNN | 0.0373 | 0.0245 |
| HOG+HSV+LBP+BoVW | Naive Bayes | 0.1549 | 0.1741 |
| HOG+HSV+LBP+BoVW | Decision Tree | 0.0412 | 0.0389 |

Mô hình được chọn là **HSV+BoVW + Gaussian Naive Bayes**, với `var_smoothing=1e-9`. Macro-F1 validation là 0.2371, cao hơn 0.1762 của HSV+KNN trong lần chạy trước.

## 6. Kết quả test của mô hình được chọn

| Chỉ số | Giá trị |
|---|---:|
| Accuracy | 0.2734 |
| Precision macro | 0.2615 |
| Recall macro | 0.2841 |
| F1 macro | 0.2518 |
| F1 weighted | 0.2659 |
| Top-5 accuracy | 0.5202 |

Các giá trị trên được đọc từ `outputs/experiments/final_test_metrics.json`. Bảng validation đầy đủ nằm ở `outputs/experiments/validation_comparison.csv`.

## 7. Phân tích lỗi

Theo `classification_report.csv`, các lớp có recall thấp nhất là corn poppy (0.000), carnation (0.000), hibiscus (0.018), sword lily (0.027) và camellia (0.028).

Các nhầm lẫn nhiều nhất trong confusion matrix là wallflower → Californian poppy (35 ảnh), watercress → wallflower (31), bishop of llandaff → wallflower (27), watercress → marigold (22) và frangipani → windflower (19). Recall trên lớp rose là 0.2384; smoke test demo với ảnh test 1143 đã phân loại đúng thành rose. Đây là thống kê trên test, không phải giải thích thị giác cho mọi lỗi.

## 8. Hạn chế và kết luận

Macro-F1 test 0.2518 tăng so với lượt HSV+KNN trước đó (0.1887), nhưng việc phân biệt 102 lớp vẫn khó. HSV+BoVW + Gaussian Naive Bayes được chọn chỉ theo macro-F1 validation; kết quả không nên được diễn giải như một hệ thống nhận diện có độ chính xác cao. SIFT BoVW và histogram màu vẫn chưa mô tả đầy đủ hình dáng hoa và ngữ cảnh.

## 9. Tái lập

Từ thư mục gốc, cài thư viện bằng `pip install -r requirements.txt`, chuẩn bị dữ liệu theo [data/README.md](data/README.md), rồi chạy:

```bash
python src/eda.py
python src/extract_features.py
python src/train_models.py
python src/app.py
```

Kết quả nêu trong báo cáo lấy từ các tệp đầu ra của pipeline trong workspace. Khi chạy lại với dữ liệu hoặc phiên bản thư viện khác, hãy thay số liệu bằng kết quả mới được sinh ra.
