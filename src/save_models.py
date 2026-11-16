import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Đường dẫn
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("1. Đang tải dữ liệu từ data/processed/...")
X_train = np.load(PROCESSED_DIR / "X_train.npy")
y_train = np.load(PROCESSED_DIR / "y_train.npy")
X_train = X_train[:, :8132]

print("2. Đang chuẩn hóa và giảm chiều (PCA)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=200, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)

print("3. Đang huấn luyện K-NN...")
# Dùng cấu hình K-NN cơ bản (k=5, distance) để chạy nhanh
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean', n_jobs=-1)
knn.fit(X_train_pca, y_train)

print("4. Đang lưu mô hình vào thư mục models/...")
joblib.dump(scaler, MODEL_DIR / "scaler_knn.pkl")
joblib.dump(pca, MODEL_DIR / "pca_knn.pkl")
joblib.dump(knn, MODEL_DIR / "knn_model.pkl")

print("✅ Đã lưu thành công 3 file (scaler, pca, knn) vào thư mục models/")