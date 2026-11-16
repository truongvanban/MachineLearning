"""Giao diện demo phân loại Oxford Flowers 102 bằng Gradio."""
from pathlib import Path
import joblib
import cv2
import gradio as gr
import numpy as np
from skimage.feature import hog

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

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

MODEL_PATH = MODEL_DIR / "best_model.joblib"
if MODEL_PATH.exists():
    saved = joblib.load(MODEL_PATH)
    pipeline = saved["pipeline"]
    feature_set = saved["feature_set"]
    model_name = saved.get("model_name", "Machine Learning")
    legacy = False
else:
    # Tương thích với bộ ba mô hình cũ nếu người dùng chưa chạy pipeline mới.
    legacy_paths = [MODEL_DIR / name for name in
                    ("scaler_knn.pkl", "pca_knn.pkl", "knn_model.pkl")]
    if not all(path.exists() for path in legacy_paths):
        raise FileNotFoundError(
            "Chưa có mô hình. Hãy chạy: python src/extract_features.py "
            "và python src/train_models.py"
        )
    scaler, pca, knn = [joblib.load(path) for path in legacy_paths]
    pipeline = (scaler, pca, knn)
    feature_set = "HOG+HSV"
    model_name = "KNN (mô hình cũ)"
    legacy = True


def extract_features(image_rgb):
    image_rgb = cv2.resize(image_rgb, (128, 128))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    feature_hog = hog(
        gray, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm="L2-Hys"
    )

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    feature_color = np.concatenate([hist_h, hist_s])
    feature_color = feature_color / (feature_color.sum() + 1e-7)
    return np.concatenate([feature_hog, feature_color]).reshape(1, -1)


def predict_flower(image):
    if image is None:
        return "Vui lòng tải lên một bức ảnh hoa."

    features = extract_features(image)
    if feature_set == "HOG":
        features = features[:, :8100]
    elif feature_set == "HSV":
        features = features[:, 8100:]

    if legacy:
        features = pipeline[0].transform(features)
        features = pipeline[1].transform(features)
        model = pipeline[2]
    else:
        model = pipeline

    prediction = int(model.predict(features)[0])
    if not 1 <= prediction <= len(CLASS_NAMES):
        return f"Mã lớp dự đoán không hợp lệ: {prediction}"

    result = f"🌸 {CLASS_NAMES[prediction - 1].title()}\nMô hình: {model_name}"
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        top_indices = np.argsort(probabilities)[-3:][::-1]
        details = [
            f"{CLASS_NAMES[int(model.classes_[i]) - 1].title()}: {probabilities[i] * 100:.1f}%"
            for i in top_indices
        ]
        result += "\n\nTop dự đoán:\n" + "\n".join(details)
    return result


demo = gr.Interface(
    fn=predict_flower,
    inputs=gr.Image(
        label="Tải ảnh hoa hoặc chụp từ camera",
        sources=["upload", "webcam", "clipboard"],
        type="numpy",
    ),
    outputs=gr.Textbox(label="Kết quả", lines=6),
    title="🌸 Nhận diện 102 loài hoa - Oxford Flowers 102",
    description=(
        "Tải ảnh hoa lên để dự đoán. Dự án sử dụng đặc trưng HOG/HSV "
        "và mô hình học máy cổ điển, không sử dụng Deep Learning."
    ),
)

if __name__ == "__main__":
    demo.launch(share=False)
