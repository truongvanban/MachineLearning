import numpy as np
import cv2
import scipy.io
from pathlib import Path
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

# Đường dẫn
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (128, 128)
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_BINS = LBP_POINTS + 2
LBP_GRID = 2
BOVW_WORDS = 64
SIFT_FEATURES = 120

print("Loading labels...")
labels = scipy.io.loadmat(DATA_DIR / "imagelabels.mat")["labels"][0]
setid = scipy.io.loadmat(DATA_DIR / "setid.mat")
train_ids = setid["trnid"][0] - 1
val_ids   = setid["valid"][0] - 1
test_ids  = setid["tstid"][0] - 1

def extract_features(img_id):
    path = DATA_DIR / "jpg" / f"image_{img_id+1:05d}.jpg"
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
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

    # LBP histogram theo lưới 2x2 để giữ lại một phần vị trí kết cấu.
    lbp = local_binary_pattern(
        img_gray, LBP_POINTS, LBP_RADIUS, method="uniform"
    )
    cell_h = lbp.shape[0] // LBP_GRID
    cell_w = lbp.shape[1] // LBP_GRID
    f_lbp = []
    for row in range(LBP_GRID):
        for col in range(LBP_GRID):
            region = lbp[
                row * cell_h:(row + 1) * cell_h,
                col * cell_w:(col + 1) * cell_w,
            ]
            hist, _ = np.histogram(region, bins=np.arange(LBP_BINS + 1))
            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-7
            f_lbp.extend(hist)

    return np.concatenate([f_hog, f_color, np.asarray(f_lbp)])


def extract_sift_descriptors(img_id, sift):
    path = DATA_DIR / "jpg" / f"image_{img_id+1:05d}.jpg"
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        return np.empty((0, 128), dtype=np.float32)
    return descriptors


def build_bovw(ids, vocabulary, sift, name, descriptors=None):
    features = []
    for index, img_id in tqdm(
        enumerate(ids), total=len(ids), desc=f"Processing {name} SIFT"
    ):
        image_descriptors = (
            descriptors[index]
            if descriptors is not None
            else extract_sift_descriptors(img_id, sift)
        )
        if len(image_descriptors):
            words = vocabulary.predict(image_descriptors)
            histogram = np.bincount(words, minlength=BOVW_WORDS).astype(np.float32)
            histogram /= histogram.sum() + 1e-7
        else:
            histogram = np.zeros(BOVW_WORDS, dtype=np.float32)
        features.append(histogram)
    return np.asarray(features)


def build_dataset(ids, labels_array, name):
    X, y = [], []
    for img_id in tqdm(ids, desc=f"Processing {name}"):
        X.append(extract_features(img_id))
    return np.array(X), labels_array[ids]

print("Extracting HOG, HSV, and LBP features...")
X_train, y_train = build_dataset(train_ids, labels, "Train")
X_val, y_val = build_dataset(val_ids, labels, "Validation")
X_test, y_test = build_dataset(test_ids, labels, "Test")

print("Building the SIFT visual vocabulary from training images...")
sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
train_descriptors = [
    extract_sift_descriptors(img_id, sift)
    for img_id in tqdm(train_ids, desc="Collecting train SIFT")
]
available_descriptors = [d for d in train_descriptors if len(d)]
if not available_descriptors:
    raise ValueError("SIFT did not find any keypoints in the training images.")
all_descriptors = np.vstack(available_descriptors)
if len(all_descriptors) < BOVW_WORDS:
    raise ValueError(
        f"Need at least {BOVW_WORDS} SIFT descriptors; found {len(all_descriptors)}."
    )
vocabulary = MiniBatchKMeans(
    n_clusters=BOVW_WORDS,
    random_state=42,
    batch_size=1024,
    n_init=3,
    max_iter=100,
)
vocabulary.fit(all_descriptors)

X_bovw_train = build_bovw(
    train_ids, vocabulary, sift, "Train", train_descriptors
)
X_bovw_val = build_bovw(val_ids, vocabulary, sift, "Validation")
X_bovw_test = build_bovw(test_ids, vocabulary, sift, "Test")
X_train = np.hstack([X_train, X_bovw_train])
X_val = np.hstack([X_val, X_bovw_val])
X_test = np.hstack([X_test, X_bovw_test])

print("Saving feature arrays...")
np.save(PROCESSED_DIR / "X_train.npy", X_train)
np.save(PROCESSED_DIR / "y_train.npy", y_train)
np.save(PROCESSED_DIR / "X_val.npy", X_val)
np.save(PROCESSED_DIR / "y_val.npy", y_val)
np.save(PROCESSED_DIR / "X_test.npy", X_test)
np.save(PROCESSED_DIR / "y_test.npy", y_test)
np.save(
    PROCESSED_DIR / "bovw_centers.npy",
    vocabulary.cluster_centers_.astype(np.float32),
)

print("Feature extraction complete: data/processed/")