import numpy as np
import cv2
import scipy.io
from pathlib import Path
from skimage.feature import hog
from tqdm import tqdm

# Đường dẫn
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (128, 128)

print("Đang đọc nhãn...")
labels = scipy.io.loadmat(DATA_DIR / "imagelabels.mat")["labels"][0]
setid = scipy.io.loadmat(DATA_DIR / "setid.mat")
train_ids = setid["trnid"][0] - 1
val_ids   = setid["valid"][0] - 1
test_ids  = setid["tstid"][0] - 1

def extract_features(img_id):
    path = DATA_DIR / "jpg" / f"image_{img_id+1:05d}.jpg"
    img = cv2.imread(str(path))
    img = cv2.resize(img, IMG_SIZE)
    
    # HOG
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f_hog = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    
    # Màu sắc
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([img_hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([img_hsv], [1], None, [16], [0, 256]).flatten()
    f_color = np.concatenate([hist_h, hist_s])
    f_color = f_color / (f_color.sum() + 1e-7) 
    
    return np.concatenate([f_hog, f_color])

def build_dataset(ids, labels_array, name):
    X, y = [], []
    for img_id in tqdm(ids, desc=f"Đang xử lý {name}"):
        X.append(extract_features(img_id))
    return np.array(X), labels_array[ids]

print("Bắt đầu trích xuất (quá trình này mất khoảng 3-5 phút)...")
X_train, y_train = build_dataset(train_ids, labels, "Train")
X_val, y_val = build_dataset(val_ids, labels, "Validation")
X_test, y_test = build_dataset(test_ids, labels, "Test")

print("Đang lưu file .npy...")
np.save(PROCESSED_DIR / "X_train.npy", X_train)
np.save(PROCESSED_DIR / "y_train.npy", y_train)
np.save(PROCESSED_DIR / "X_val.npy", X_val)
np.save(PROCESSED_DIR / "y_val.npy", y_val)
np.save(PROCESSED_DIR / "X_test.npy", X_test)
np.save(PROCESSED_DIR / "y_test.npy", y_test)

print("✅ Đã trích xuất và lưu xong toàn bộ dữ liệu vào data/processed/")