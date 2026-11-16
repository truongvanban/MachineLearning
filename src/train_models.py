"""So sánh đặc trưng HOG/HSV và mô hình cổ điển cho Oxford Flowers 102.

Chạy từ thư mục gốc: python src/train_models.py
Yêu cầu đã chạy: python src/extract_features.py
"""
from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, top_k_accuracy_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs" / "experiments"
HOG_DIM = 8100
N_CLASSES = 102
RANDOM_STATE = 42


def load_data():
    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    arrays = {}
    for name in required:
        path = DATA_DIR / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Thiếu {path}. Hãy chạy: python src/extract_features.py"
            )
        arrays[name] = np.load(path)
    for split in ("train", "val", "test"):
        X, y = arrays[f"X_{split}"], arrays[f"y_{split}"]
        if len(X) != len(y):
            raise ValueError(f"Số mẫu X_{split} và y_{split} không bằng nhau.")
        if X.ndim != 2 or X.shape[1] != HOG_DIM + 32:
            raise ValueError(
                f"X_{split} có shape {X.shape}; mong đợi (n, 8132). "
                "Hãy trích xuất lại bằng src/extract_features.py."
            )
        if not np.isfinite(X).all():
            raise ValueError(f"X_{split} chứa NaN hoặc Inf.")
        if y.min() < 1 or y.max() > N_CLASSES:
            raise ValueError(f"Nhãn y_{split} phải nằm trong khoảng 1..102.")
    return arrays


def make_features(X, feature_name):
    if feature_name == "HOG":
        return X[:, :HOG_DIM]
    if feature_name == "HSV":
        return X[:, HOG_DIM:]
    return X


def score_model(name, model, X, y):
    pred = model.predict(X)
    result = {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
    }
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        result["top5_accuracy"] = top_k_accuracy_score(
            y, prob, k=5, labels=model.classes_
        )
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        result["top5_accuracy"] = top_k_accuracy_score(
            y, decision, k=5, labels=model.classes_
        )
    else:
        result["top5_accuracy"] = np.nan
    return result, pred


def main():
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Validation được dùng để chọn mô hình; test chỉ dùng đánh giá cuối.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    experiments = []
    candidates = []

    for feature_name in ("HOG", "HSV", "HOG+HSV"):
        Xtr = make_features(X_train, feature_name)
        Xv = make_features(X_val, feature_name)
        Xt = make_features(X_test, feature_name)

        # Baseline KNN: giữ vai trò mốc so sánh.
        knn = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=min(200, Xtr.shape[0] - 1, Xtr.shape[1]), random_state=RANDOM_STATE)),
            ("model", KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=-1)),
        ])
        knn.fit(Xtr, y_train)
        result, _ = score_model(f"{feature_name} + KNN baseline", knn, Xv, y_val)
        experiments.append(result)
        candidates.append((result["f1_macro"], f"{feature_name} + KNN", feature_name, knn))

        # SVM tuyến tính và RBF, chọn tham số bằng CV trên train.
        svm_pipe = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=min(200, Xtr.shape[0] - 1), random_state=RANDOM_STATE)),
            ("model", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        ])
        grid = GridSearchCV(
            svm_pipe,
            {
                "model__kernel": ["linear", "rbf"],
                "model__C": [0.1, 1, 10],
                "model__gamma": ["scale"] if feature_name == "HSV" else ["scale", 0.01],
            },
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        grid.fit(Xtr, y_train)
        result, _ = score_model(f"{feature_name} + SVM", grid.best_estimator_, Xv, y_val)
        result["best_params"] = json.dumps(grid.best_params_, ensure_ascii=False)
        experiments.append(result)
        candidates.append((result["f1_macro"], f"{feature_name} + SVM", feature_name, grid.best_estimator_))
        print(f"{feature_name}: best SVM={grid.best_params_}; val macro-F1={result['f1_macro']:.4f}")

    # Chọn duy nhất theo validation, sau đó fit lại train+validation.
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_val_f1, best_name, best_feature, best_pipeline = candidates[0]
    X_trainval = np.vstack([
        make_features(X_train, best_feature),
        make_features(X_val, best_feature),
    ])
    y_trainval = np.concatenate([y_train, y_val])
    best_pipeline.fit(X_trainval, y_trainval)

    Xtest_final = make_features(X_test, best_feature)
    test_result, y_pred = score_model(best_name, best_pipeline, Xtest_final, y_test)
    test_result["selected_by"] = "validation macro-F1"
    test_result["validation_macro_f1"] = best_val_f1
    test_result["feature_set"] = best_feature

    # Lưu toàn bộ bảng validation và kết quả test của mô hình đã chọn.
    pd.DataFrame(experiments).to_csv(OUT_DIR / "validation_comparison.csv", index=False)
    with (OUT_DIR / "final_test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_result, f, ensure_ascii=False, indent=2)
    report = classification_report(
        y_test, y_pred, labels=np.arange(1, 103), zero_division=0, output_dict=True
    )
    pd.DataFrame(report).transpose().to_csv(OUT_DIR / "classification_report.csv")

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(1, 103))
    np.save(OUT_DIR / "confusion_matrix.npy", cm)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.set_title(f"Confusion matrix - {best_name}")
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    fig.colorbar(im, ax=ax, label="Số ảnh")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    joblib.dump(
        {"pipeline": best_pipeline, "feature_set": best_feature,
         "feature_dim": int(X_trainval.shape[1]), "model_name": best_name},
        MODEL_DIR / "best_model.joblib",
    )
    print("\nMô hình tốt nhất:", best_name)
    print("Validation macro-F1:", round(best_val_f1, 4))
    print("Test metrics:", json.dumps(test_result, ensure_ascii=False, indent=2))
    print("Classification report:", OUT_DIR / "classification_report.csv")
    print(f"Thời gian: {(time.time() - start) / 60:.1f} phút")


if __name__ == "__main__":
    main()
