# BÁO CÁO ĐỀ TÀI: PHÂN LOẠI 102 LOÀI HOA BẰNG ĐẶC TRƯNG THỦ CÔNG VÀ HỌC MÁY CỔ ĐIỂN

> Đây là khung báo cáo có thể tái lập. Những vị trí ghi **[BỔ SUNG SAU KHI CHẠY]** phải được điền từ các tệp kết quả trong `outputs/`; không sử dụng số liệu minh họa như kết quả thật.

## 1. Giới thiệu
Bài toán nhận dạng ảnh đầu vào thuộc một trong 102 lớp hoa của bộ Oxford Flowers 102. Phạm vi đề tài sử dụng xử lý ảnh và học máy cổ điển, không sử dụng Deep Learning.

## 2. Dữ liệu
Oxford Flowers 102 gồm 8.189 ảnh thuộc 102 lớp. Theo split chính thức, train có 1.020 ảnh, validation có 1.020 ảnh và test có 6.149 ảnh. Nhãn và ID chia tập được đọc từ `imagelabels.mat` và `setid.mat`.

**Kiểm tra dữ liệu:** chạy `python src/eda.py`. Bảng phân phối lớp và thông tin dataset được lưu trong `outputs/eda/dataset_summary.json`.

## 3. Tiền xử lý và trích xuất đặc trưng
- Resize ảnh về 128 × 128 pixel.
- HOG: ảnh xám, 9 hướng gradient, cell 8 × 8, block 2 × 2.
- HSV: histogram 16 bins cho kênh H và 16 bins cho kênh S, chuẩn hóa L1.
- LBP: histogram uniform với 8 điểm lân cận, bán kính 1; histogram riêng trên lưới 2 × 2 (40 chiều).
- SIFT BoVW: tối đa 120 keypoint/ảnh; vocabulary 64 từ được fit trên descriptor từ tập train; histogram được chuẩn hóa L1.
- So sánh bảy tập đặc trưng: HOG, HSV, HOG+HSV, HOG+HSV+LBP, BoVW, HSV+BoVW và HOG+HSV+LBP+BoVW.
- Vector kết hợp đầy đủ có 8.236 chiều (8.100 HOG + 32 HSV + 40 LBP + 64 BoVW).

## 4. Mô hình và thiết kế thực nghiệm
- Ba thuật toán được so sánh: KNN, Gaussian Naive Bayes và Decision Tree.
- KNN tìm số láng giềng và trọng số; Gaussian Naive Bayes dùng `var_smoothing=1e-9`; Decision Tree tìm tiêu chí chia nhánh, độ sâu và số mẫu lá tối thiểu, với `max_features="sqrt"` để giảm thời gian trên vector HOG lớn.
- KNN dùng StandardScaler+PCA, ngoại trừ các tập có BoVW dùng chuẩn hóa L1 và cosine distance. Gaussian Naive Bayes dùng StandardScaler, không PCA; Decision Tree dùng trực tiếp đặc trưng.
- Dùng StratifiedKFold 3 folds trên train để chọn siêu tham số.
- Validation macro-F1 dùng để chọn cấu hình. Sau khi chọn, mô hình được fit lại trên train + validation và đánh giá một lần trên test.

## 5. Kết quả
**[BỔ SUNG SAU KHI CHẠY]** Lấy bảng so sánh validation từ `outputs/experiments/validation_comparison.csv` và metric cuối từ `outputs/experiments/final_test_metrics.json`.

Bắt buộc báo cáo Accuracy, Precision macro, Recall macro, F1 macro, F1 weighted và Top-5 accuracy (nếu mô hình hỗ trợ). Không chỉ dùng Accuracy vì các lớp có số lượng ảnh không hoàn toàn bằng nhau.

## 6. Phân tích lỗi
**[BỔ SUNG SAU KHI CHẠY]** Dùng `outputs/experiments/confusion_matrix.png` và `classification_report.csv` để xác định lớp dễ nhầm, lớp có recall thấp và các cặp nhãn nhầm lẫn. Chọn một số ảnh dự đoán sai để giải thích dựa trên màu sắc, hình dạng, góc chụp, nền và ánh sáng.

## 7. Demo
Chạy `python src/app.py` sau khi đã tạo `models/best_model.joblib`. Giao diện Gradio cho phép tải ảnh lên và hiển thị lớp dự đoán.

## 8. Hạn chế
- Mỗi lớp chỉ có 10 ảnh trong tập train chính thức; số mẫu huấn luyện nhỏ so với 102 lớp.
- HOG và histogram màu là đặc trưng thủ công, chưa mô tả đầy đủ ngữ nghĩa và cấu trúc phức tạp của hoa.
- Kết quả có thể thay đổi theo tham số, cách chuẩn hóa và môi trường chạy.
- Không kết luận mô hình tốt nhất trước khi chạy lại toàn bộ pipeline trên dữ liệu thật.

## 9. Kết luận
**[BỔ SUNG SAU KHI CHẠY]** Nêu cấu hình tốt nhất theo validation, kết quả test, ưu/nhược điểm và hướng cải thiện trong phạm vi học máy cổ điển.

## 10. Tái lập
Môi trường, phiên bản thư viện, random_state, tham số và lệnh chạy được ghi trong README và các tệp kết quả. Dữ liệu gốc không đưa lên Git; người chạy cần tự tải từ nguồn Oxford.
