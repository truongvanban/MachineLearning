import gradio as gr
import cv2
import numpy as np
import joblib
from pathlib import Path
from skimage.feature import hog

# 1. Định nghĩa đường dẫn
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"

# 2. DANH SÁCH 102 LOÀI HOA
CLASS_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold",
    "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle",
    "snapdragon", "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower", "giant white arum lily",
    "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth",
    "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william",
    "carnation", "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip", "lenten rose",
    "barbeton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue",
    "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
    "petunia", "wild pansy", "primula", "sunflower", "pelargonium",
    "bishop of llandaff", "gaura", "geranium", "orange dahlia", "pink-yellow dahlia",
    "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush", "californian poppy",
    "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy",
    "gazania", "azalea", "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus", "toad lily", "anthurium",
    "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
    "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily",
    "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea",
    "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily"
]

# 3. Load mô hình
print("Đang tải mô hình...")
scaler = joblib.load(MODEL_DIR / "scaler_knn.pkl")
pca = joblib.load(MODEL_DIR / "pca_knn.pkl")
knn = joblib.load(MODEL_DIR / "knn_model.pkl")
print("Tải mô hình thành công!")

# 4. Hàm trích xuất đặc trưng
def extract_features_from_image(img_rgb):
    img = cv2.resize(img_rgb, (128, 128))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # HOG
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    f_hog = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    
    # Màu sắc
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([img_hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([img_hsv], [1], None, [16], [0, 256]).flatten()
    f_color = np.concatenate([hist_h, hist_s])
    f_color = f_color / (f_color.sum() + 1e-7)
    
    return np.concatenate([f_hog, f_color]).reshape(1, -1)

# 5. Hàm dự đoán
def predict_flower(image):
    if image is None:
        return "Vui lòng tải lên hoặc chụp một bức ảnh!"
    
    features = extract_features_from_image(image)
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    
    prediction = knn.predict(features_pca)[0]
    
    # Tính độ tự tin chuẩn
    probabilities = knn.predict_proba(features_pca)[0]
    class_idx = np.where(knn.classes_ == prediction)[0][0]
    confidence = probabilities[class_idx] * 100
    
    flower_name = CLASS_NAMES[prediction - 1]
    flower_name_formatted = flower_name.title()
    
    return f"🌸 Loài hoa: {flower_name_formatted} \n\n(Độ tự tin: {confidence:.2f}%)"

# 6. Xây dựng giao diện Gradio (Trở lại như ban đầu + Camera tĩnh)
title = "🌸 Phân Loại Hoa (Oxford Flowers 102) - K-NN"
description = "Tải lên một bức ảnh hoặc chụp từ Camera, mô hình K-NN (sử dụng đặc trưng HOG và Màu sắc) sẽ dự đoán tên loài hoa đó."

interface = gr.Interface(
    fn=predict_flower, 
    # sources=["upload", "webcam"] cho phép vừa tải ảnh lên, vừa bật camera chụp tĩnh
        inputs=gr.Image(
        label="Tải ảnh, Chụp từ Camera hoặc Dán (Ctrl+V)", 
        sources=["upload", "webcam", "clipboard"]
    ),
    outputs=gr.Textbox(label="Kết quả dự đoán", lines=3),
    title=title,
    description=description,
    examples=["../data/raw/jpg/image_00001.jpg", "../data/raw/jpg/image_00500.jpg"]
)

if __name__ == "__main__":
    interface.launch(share=False)