"""Giao diện demo phân loại Oxford Flowers 102 bằng Gradio."""
from pathlib import Path
import joblib
import cv2
import gradio as gr
import numpy as np
from skimage.feature import hog, local_binary_pattern

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
HOG_DIM = 8100
HSV_DIM = 32
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_BINS = LBP_POINTS + 2
LBP_GRID = 2
BOVW_WORDS = 64
SIFT_FEATURES = 120
BASE_FEATURE_DIM = HOG_DIM + HSV_DIM + LBP_BINS * LBP_GRID * LBP_GRID

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
bovw_centers = None
if MODEL_PATH.exists():
    saved = joblib.load(MODEL_PATH)
    pipeline = saved["pipeline"]
    feature_set = saved["feature_set"]
    model_name = saved.get("model_name", "Machine Learning")
    bovw_centers = saved.get("bovw_centers")
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


def extract_bovw(gray, centers):
    if centers is None or centers.shape != (BOVW_WORDS, 128):
        raise ValueError(
            "Mô hình BoVW thiếu vocabulary hợp lệ; hãy chạy lại "
            "python src/extract_features.py và python src/train_models.py."
        )
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    _, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros(BOVW_WORDS, dtype=np.float32)
    distances = np.sum(
        (descriptors[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    words = np.argmin(distances, axis=1)
    histogram = np.bincount(words, minlength=BOVW_WORDS).astype(np.float32)
    return histogram / (histogram.sum() + 1e-7)


def extract_features(image_rgb, include_bovw=False):
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
    lbp = local_binary_pattern(
        gray, LBP_POINTS, LBP_RADIUS, method="uniform"
    )
    cell_h = lbp.shape[0] // LBP_GRID
    cell_w = lbp.shape[1] // LBP_GRID
    feature_lbp = []
    for row in range(LBP_GRID):
        for col in range(LBP_GRID):
            region = lbp[
                row * cell_h:(row + 1) * cell_h,
                col * cell_w:(col + 1) * cell_w,
            ]
            hist, _ = np.histogram(region, bins=np.arange(LBP_BINS + 1))
            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-7
            feature_lbp.extend(hist)
    features = np.concatenate(
        [feature_hog, feature_color, np.asarray(feature_lbp)]
    )
    if include_bovw:
        features = np.concatenate([features, extract_bovw(gray, bovw_centers)])
    return features.reshape(1, -1)


def predict_flower(image):
    if image is None:
        return "Vui lòng tải lên một bức ảnh hoa."

    features = extract_features(image, include_bovw="BoVW" in feature_set)
    if feature_set == "HOG":
        features = features[:, :HOG_DIM]
    elif feature_set == "HSV":
        features = features[:, HOG_DIM:HOG_DIM + HSV_DIM]
    elif feature_set == "HOG+HSV":
        features = features[:, :HOG_DIM + HSV_DIM]
    elif feature_set == "HOG+HSV+LBP":
        features = features[:, :BASE_FEATURE_DIM]
    elif feature_set == "BoVW":
        features = features[:, BASE_FEATURE_DIM:BASE_FEATURE_DIM + BOVW_WORDS]
    elif feature_set == "HSV+BoVW":
        features = np.hstack([
            features[:, HOG_DIM:HOG_DIM + HSV_DIM],
            features[:, BASE_FEATURE_DIM:BASE_FEATURE_DIM + BOVW_WORDS],
        ])
    elif feature_set != "HOG+HSV+LBP+BoVW":
        raise ValueError(f"Tập đặc trưng không được hỗ trợ: {feature_set}")

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
        "Tải ảnh hoa lên để dự đoán. Dự án sử dụng HOG/HSV/LBP/SIFT BoVW "
        "và mô hình học máy cổ điển, không sử dụng Deep Learning."
    ),
)

if __name__ == "__main__":
    demo.launch(share=False)
