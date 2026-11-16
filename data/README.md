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

Tập train gồm 1.020 ảnh, validation 1.020 ảnh và test 6.149 ảnh theo split chính thức. ID trong `setid.mat` bắt đầu từ 1; mã nguồn chuyển về chỉ số Python bằng cách trừ 1 khi truy cập nhãn.
