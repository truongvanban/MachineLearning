"""So sánh đặc trưng HOG/HSV/LBP/BoVW và mô hình cổ điển cho Oxford Flowers 102.

Chạy từ thư mục gốc: python src/train_models.py
Yêu cầu đã chạy: python src/extract_features.py
"""
from pathlib import Path
import json
import tempfile
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
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs" / "experiments"
HOG_DIM = 8100
HSV_DIM = 32
LBP_POINTS = 8
LBP_BINS = LBP_POINTS + 2
LBP_GRID = 2
LBP_DIM = LBP_BINS * LBP_GRID * LBP_GRID
BOVW_WORDS = 64
BOVW_DIM = BOVW_WORDS
BASE_FEATURE_DIM = HOG_DIM + HSV_DIM + LBP_DIM
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
        expected_dim = BASE_FEATURE_DIM + BOVW_DIM
        if X.ndim != 2 or X.shape[1] != expected_dim:
            raise ValueError(
                f"X_{split} có shape {X.shape}; mong đợi (n, {expected_dim}). "
                "Hãy chạy lại python src/extract_features.py để tạo HOG+HSV+LBP+BoVW."
            )
        if not np.isfinite(X).all():
            raise ValueError(f"X_{split} chứa NaN hoặc Inf.")
        if y.min() < 1 or y.max() > N_CLASSES:
            raise ValueError(f"Nhãn y_{split} phải nằm trong khoảng 1..102.")
    centers_path = DATA_DIR / "bovw_centers.npy"
    if not centers_path.exists():
        raise FileNotFoundError(
            f"Thiếu {centers_path}. Hãy chạy: python src/extract_features.py"
        )
    centers = np.load(centers_path)
    if centers.shape != (BOVW_WORDS, 128) or not np.isfinite(centers).all():
        raise ValueError(
            f"Vocabulary BoVW có shape {centers.shape}; mong đợi "
            f"({BOVW_WORDS}, 128). Hãy chạy lại bước trích xuất đặc trưng."
        )
    arrays["bovw_centers"] = centers
    return arrays


def make_features(X, feature_name):
    if feature_name == "HOG":
        return X[:, :HOG_DIM]
    if feature_name == "HSV":
        return X[:, HOG_DIM:HOG_DIM + HSV_DIM]
    if feature_name == "HOG+HSV":
        return X[:, :HOG_DIM + HSV_DIM]
    if feature_name == "HOG+HSV+LBP":
        return X[:, :BASE_FEATURE_DIM]
    if feature_name == "BoVW":
        return X[:, BASE_FEATURE_DIM:BASE_FEATURE_DIM + BOVW_DIM]
    if feature_name == "HSV+BoVW":
        return np.hstack([
            X[:, HOG_DIM:HOG_DIM + HSV_DIM],
            X[:, BASE_FEATURE_DIM:BASE_FEATURE_DIM + BOVW_DIM],
        ])
    if feature_name == "HOG+HSV+LBP+BoVW":
        return X
    raise ValueError(f"Tập đặc trưng không được hỗ trợ: {feature_name}")


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
    cache_dir = tempfile.TemporaryDirectory(prefix="flower102-pipeline-")
    memory = joblib.Memory(location=cache_dir.name, verbose=0)
    experiments = []
    candidates = []

    for feature_name in (
        "HOG", "HSV", "HOG+HSV", "HOG+HSV+LBP",
        "BoVW", "HSV+BoVW", "HOG+HSV+LBP+BoVW",
    ):
        Xtr = make_features(X_train, feature_name)
        Xv = make_features(X_val, feature_name)
        n_components = min(200, Xtr.shape[0] - 1, Xtr.shape[1])
        if "BoVW" in feature_name:
            knn_pipeline = Pipeline([
                ("normalize", Normalizer()),
                ("model", KNeighborsClassifier(metric="cosine", n_jobs=-1)),
            ], memory=memory)
        else:
            knn_pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
                ("model", KNeighborsClassifier(n_jobs=-1)),
            ], memory=memory)

        model_specs = (
            (
                "KNN",
                knn_pipeline,
                {
                    "model__n_neighbors": [3, 5, 7],
                    "model__weights": ["uniform", "distance"],
                },
            ),
            (
                "Naive Bayes",
                Pipeline([
                    ("scale", StandardScaler()),
                    ("model", GaussianNB()),
                ], memory=memory),
                # A full parameter sweep on these features selected 1e-9 everywhere.
                {"model__var_smoothing": [1e-9]},
            ),
            (
                "Decision Tree",
                Pipeline([
                    ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
                ]),
                {
                    "model__criterion": ["gini", "entropy"],
                    "model__max_depth": [None, 30],
                    "model__min_samples_leaf": [1, 2],
                    "model__max_features": ["sqrt"],
                },
            ),
        )

        for model_label, pipeline, param_grid in model_specs:
            model_name = f"{feature_name} + {model_label}"
            grid = GridSearchCV(
                pipeline,
                param_grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=1,
                refit=True,
            )
            model_start = time.time()
            print(f"Training {model_name} with cross-validation...", flush=True)
            grid.fit(Xtr, y_train)
            result, _ = score_model(model_name, grid.best_estimator_, Xv, y_val)
            result["best_params"] = json.dumps(grid.best_params_, ensure_ascii=False)
            experiments.append(result)
            candidates.append((
                result["f1_macro"], model_name, feature_name, grid.best_estimator_
            ))
            print(
                f"{model_name}: best params={grid.best_params_}; "
                f"val macro-F1={result['f1_macro']:.4f}; "
                f"elapsed={time.time() - model_start:.1f}s",
                flush=True,
            )

    # Chọn duy nhất theo validation, sau đó fit lại train+validation.
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_val_f1, best_name, best_feature, best_pipeline = candidates[0]
    best_pipeline.memory = None
    X_trainval = np.vstack([
        make_features(X_train, best_feature),
        make_features(X_val, best_feature),
    ])
    y_trainval = np.concatenate([y_train, y_val])
    best_pipeline.fit(X_trainval, y_trainval)
    cache_dir.cleanup()

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
         "feature_dim": int(X_trainval.shape[1]), "model_name": best_name,
         "bovw_centers": data["bovw_centers"]},
        MODEL_DIR / "best_model.joblib",
    )

    # Lưu riêng mô hình KNN tốt nhất để app dùng làm dự đoán chính.
    # Naive Bayes và Decision Tree chỉ chạy để so sánh, không lưu riêng.
    knn_candidates = [c for c in candidates if c[1].endswith(" + KNN")]
    knn_candidates.sort(key=lambda item: item[0], reverse=True)
    if knn_candidates:
        best_knn_f1, best_knn_name, best_knn_feature, best_knn_pipeline = knn_candidates[0]
        best_knn_pipeline.memory = None
        X_knn_trainval = np.vstack([
            make_features(X_train, best_knn_feature),
            make_features(X_val, best_knn_feature),
        ])
        y_knn_trainval = np.concatenate([y_train, y_val])
        best_knn_pipeline.fit(X_knn_trainval, y_knn_trainval)
        joblib.dump(
            {"pipeline": best_knn_pipeline, "feature_set": best_knn_feature,
             "model_name": best_knn_name, "bovw_centers": data["bovw_centers"]},
            MODEL_DIR / "knn_model.joblib",
        )
        print("KNN model saved:", best_knn_name)

    print("\nBest model:", best_name)
    print("Validation macro-F1:", round(best_val_f1, 4))
    print("Test metrics:", json.dumps(test_result, ensure_ascii=False, indent=2))
    print("Classification report:", OUT_DIR / "classification_report.csv")
    print(f"Elapsed time: {(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
