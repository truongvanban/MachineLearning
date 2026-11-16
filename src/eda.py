"""Kiểm tra dữ liệu và tạo báo cáo EDA cho Oxford Flowers 102."""
from pathlib import Path
import json
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    labels_path = DATA_DIR / "imagelabels.mat"
    split_path = DATA_DIR / "setid.mat"
    images_dir = DATA_DIR / "jpg"
    for path in (labels_path, split_path, images_dir):
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {path}. Hãy tải và giải nén Oxford Flowers 102 theo README.md."
            )

    labels = scipy.io.loadmat(labels_path)["labels"].ravel().astype(int)
    split = scipy.io.loadmat(split_path)
    splits = {
        "train": split["trnid"].ravel().astype(int),
        "validation": split["valid"].ravel().astype(int),
        "test": split["tstid"].ravel().astype(int),
    }

    # File setid.mat và imagelabels.mat dùng ID ảnh bắt đầu từ 1.
    all_ids = np.concatenate(list(splits.values()))
    if all_ids.min() != 1 or all_ids.max() != len(labels):
        raise ValueError("ID trong setid.mat không khớp số lượng nhãn trong imagelabels.mat.")
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("Các tập train/validation/test bị trùng ID ảnh.")
    if set(all_ids.tolist()) != set(range(1, len(labels) + 1)):
        raise ValueError("Có ảnh không thuộc bất kỳ tập dữ liệu nào.")

    summary = {}
    for name, ids in splits.items():
        y = labels[ids - 1]
        counts = np.bincount(y, minlength=103)[1:103]
        summary[name] = {
            "images": int(len(ids)),
            "classes_present": int(np.count_nonzero(counts)),
            "min_images_per_class": int(counts.min()),
            "max_images_per_class": int(counts.max()),
            "class_counts": counts.tolist(),
        }
        print(
            f"{name:10s}: {len(ids):4d} ảnh | "
            f"{np.count_nonzero(counts):3d}/102 lớp | "
            f"min={counts.min()}, max={counts.max()}"
        )

    missing = [i for i in range(1, len(labels) + 1)
               if not (images_dir / f"image_{i:05d}.jpg").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Thiếu {len(missing)} ảnh; ví dụ ID: {missing[:10]}"
        )

    with (OUT_DIR / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (name, info) in zip(axes, summary.items()):
        ax.bar(np.arange(1, 103), info["class_counts"])
        ax.set_title(f"{name}: {info['images']} ảnh")
        ax.set_xlabel("Mã lớp")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Số ảnh")
    fig.suptitle("Phân phối số ảnh theo lớp - Oxford Flowers 102")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "class_distribution.png", dpi=160)
    plt.close(fig)
    print(f"Đã lưu kết quả EDA tại: {OUT_DIR}")


if __name__ == "__main__":
    main()
