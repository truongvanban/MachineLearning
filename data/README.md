# Chuẩn bị dữ liệu Oxford Flowers 102

Tải bộ dữ liệu gốc từ trang Oxford Visual Geometry Group:

- Ảnh: `102flowers.tgz`
- Nhãn: `imagelabels.mat`
- Chia tập: `setid.mat`

Giải nén và đặt đúng cấu trúc:

```
data/
└── raw/
    ├── jpg/
    │   ├── image_00001.jpg
    │   └── ...
    ├── imagelabels.mat
    └── setid.mat
```

Dữ liệu gốc và dữ liệu đặc trưng không được commit lên Git vì dung lượng lớn. Sau khi chuẩn bị dữ liệu, chạy:

```bash
python src/eda.py
python src/extract_features.py
python src/train_models.py
```

Mã nguồn trích xuất HOG, HSV, histogram LBP theo lưới 2×2 và SIFT BoVW với vocabulary 64 từ. Vocabulary được học từ descriptor của tập train; các vector train/validation/test dùng chung vocabulary đó. Nếu thay đổi cách trích xuất hoặc nâng cấp mã nguồn, hãy chạy lại `python src/extract_features.py` trước khi huấn luyện; các tệp `.npy` từ phiên bản cũ không còn đủ chiều đặc trưng.

Tập train gồm 1.020 ảnh, validation 1.020 ảnh và test 6.149 ảnh theo split chính thức. ID trong `setid.mat` bắt đầu từ 1; mã nguồn chuyển về chỉ số Python bằng cách trừ 1 khi truy cập nhãn.
