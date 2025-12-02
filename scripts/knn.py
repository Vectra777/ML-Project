"""
KNN training and evaluation with cached CNN features and metrics plots.
- Uses MobileNetV2 (global average pooled) to keep feature dim small.
- Caches train/test features to disk to avoid recomputing on every run.
- Saves confusion matrices and precision (train vs test) plots.
"""

import pickle
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ================= CONFIG =================
IMAGE_SIZE = (64, 64)
CNN_SIZE = (224, 224)
FEATURE_BATCH = 32  # images per batch for feature extraction
SCALE_BATCH = 512   # rows per batch when scaling
N_NEIGHBORS = 3
USE_RAW_PIXELS = False  # keep False for faster KNN; set True to append 64x64 grayscale pixels

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
TRAIN_FEATURES_PATH = ARTIFACTS_DIR / "knn_train_features.dat"
TEST_FEATURES_PATH = ARTIFACTS_DIR / "knn_test_features.dat"
TRAIN_SCALED_PATH = ARTIFACTS_DIR / "knn_train_scaled.dat"
TEST_SCALED_PATH = ARTIFACTS_DIR / "knn_test_scaled.dat"
SCALER_PATH = ARTIFACTS_DIR / "knn_scaler.pkl"
MODEL_PATH = ARTIFACTS_DIR / "knn_model.pkl"
REPORT_PATH = ARTIFACTS_DIR / "knn_report.txt"
CONFUSION_TRAIN_IMG = ARTIFACTS_DIR / "knn_confusion_train.png"
CONFUSION_TEST_IMG = ARTIFACTS_DIR / "knn_confusion_test.png"
CONFUSION_TEST_CSV = ARTIFACTS_DIR / "knn_confusion_test.csv"
PRECISION_IMG = ARTIFACTS_DIR / "knn_precision.png"

cnn_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
CNN_FEATURES = int(np.prod(cnn_model.output_shape[1:]))
PIXEL_FEATURES = IMAGE_SIZE[0] * IMAGE_SIZE[1] if USE_RAW_PIXELS else 0
FEATURE_DIM = CNN_FEATURES + PIXEL_FEATURES


# ================= HELPERS =================
def iter_chunks(n_items: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        yield start, end


def _existing_memmap_valid(path: Path, count: int) -> bool:
    if not path.exists():
        return False
    expected_bytes = count * FEATURE_DIM * np.dtype(np.float32).itemsize
    return path.stat().st_size == expected_bytes


# ================= FEATURE EXTRACTION =================
def extract_batch_features(paths: List[str]) -> np.ndarray:
    batch_size = len(paths)
    pixel_feats = None
    if USE_RAW_PIXELS:
        pixel_feats = np.empty((batch_size, PIXEL_FEATURES), dtype=np.float32)

    cnn_input = np.empty((batch_size, *CNN_SIZE, 3), dtype=np.float32)
    for i, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Impossible de lire l'image : {path}")
        if USE_RAW_PIXELS:
            gray = cv2.cvtColor(cv2.resize(img, IMAGE_SIZE), cv2.COLOR_BGR2GRAY)
            pixel_feats[i] = gray.flatten()
        resized = cv2.resize(img, CNN_SIZE)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        cnn_input[i] = resized

    cnn_input = preprocess_input(cnn_input)
    cnn_feats = cnn_model.predict(cnn_input, verbose=0)
    if cnn_feats.ndim > 2:
        cnn_feats = cnn_feats.reshape(batch_size, -1)

    if USE_RAW_PIXELS:
        return np.hstack([pixel_feats, cnn_feats])
    return cnn_feats.astype(np.float32)


def build_feature_store(df, split: str, batch_size: int = FEATURE_BATCH, force: bool = False) -> Path:
    df = df.reset_index(drop=True)
    count = len(df)
    path = TRAIN_FEATURES_PATH if split == "train" else TEST_FEATURES_PATH
    if path.exists() and not force and _existing_memmap_valid(path, count):
        print(f"[FEATS] Reusing cached {split} features at {path}")
        return path
    if path.exists():
        path.unlink()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    features = np.memmap(path, dtype=np.float32, mode="w+", shape=(count, FEATURE_DIM))
    print(f"[FEATS] Building {split} features ({count} samples) in batches of {batch_size}...")
    for idx, (start, end) in enumerate(iter_chunks(count, batch_size), start=1):
        batch_paths = df["path"].iloc[start:end].tolist()
        feats = extract_batch_features(batch_paths)
        features[start:end] = feats
        if idx % 10 == 0 or end == count:
            print(f"  - Processed {end}/{count}")
    features.flush()
    return path


# ================= SCALING =================
def fit_scaler(features_path: Path, sample_count: int) -> StandardScaler:
    scaler = StandardScaler()
    mmap = np.memmap(features_path, dtype=np.float32, mode="r", shape=(sample_count, FEATURE_DIM))
    for start, end in iter_chunks(sample_count, SCALE_BATCH):
        scaler.partial_fit(mmap[start:end])
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    return scaler


def transform_split(features_path: Path, sample_count: int, scaler: StandardScaler, split: str) -> Path:
    out_path = TRAIN_SCALED_PATH if split == "train" else TEST_SCALED_PATH
    if out_path.exists():
        out_path.unlink()
    scaled = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(sample_count, FEATURE_DIM))
    src = np.memmap(features_path, dtype=np.float32, mode="r", shape=(sample_count, FEATURE_DIM))

    for start, end in iter_chunks(sample_count, SCALE_BATCH):
        scaled[start:end] = scaler.transform(src[start:end])
    scaled.flush()
    return out_path


# ================= TRAIN & EVAL =================
def train_knn(X_train_path: Path, y_train: np.ndarray) -> KNeighborsClassifier:
    X_train = np.memmap(X_train_path, dtype=np.float32, mode="r", shape=(len(y_train), FEATURE_DIM))
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=-1, weights="distance")
    print(f"[TRAIN] Fitting KNN on {len(y_train)} samples with {FEATURE_DIM} features...")
    knn.fit(X_train, y_train)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(knn, f)
    return knn


def evaluate(
    knn: KNeighborsClassifier,
    X_train_path: Path,
    y_train: np.ndarray,
    X_test_path: Path,
    y_test: np.ndarray,
    classes: List[str],
) -> None:
    X_train = np.memmap(X_train_path, dtype=np.float32, mode="r", shape=(len(y_train), FEATURE_DIM))
    X_test = np.memmap(X_test_path, dtype=np.float32, mode="r", shape=(len(y_test), FEATURE_DIM))

    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    prec_train = precision_score(y_train, y_pred_train, labels=classes, average="macro", zero_division=0)
    prec_test = precision_score(y_test, y_pred_test, labels=classes, average="macro", zero_division=0)

    report = classification_report(y_test, y_pred_test, labels=classes, target_names=classes, digits=4)
    cm_train = confusion_matrix(y_train, y_pred_train, labels=classes)
    cm_test = confusion_matrix(y_test, y_pred_test, labels=classes)

    with open(REPORT_PATH, "w") as f:
        f.write(f"Train accuracy: {acc_train:.4f}\n")
        f.write(f"Test accuracy:  {acc_test:.4f}\n")
        f.write(f"Train precision (macro): {prec_train:.4f}\n")
        f.write(f"Test precision  (macro): {prec_test:.4f}\n\n")
        f.write(report)

    pd.DataFrame(cm_test, index=classes, columns=classes).to_csv(CONFUSION_TEST_CSV)
    _plot_confusion(cm_train, classes, CONFUSION_TRAIN_IMG, title="Train confusion matrix")
    _plot_confusion(cm_test, classes, CONFUSION_TEST_IMG, title="Test confusion matrix")
    _plot_precision(prec_train, prec_test, PRECISION_IMG)

    print(f"[EVAL] Train acc: {acc_train:.4f} | Test acc: {acc_test:.4f}")
    print(f"[EVAL] Train precision (macro): {prec_train:.4f} | Test precision (macro): {prec_test:.4f}")
    print(f"[EVAL] Classification report saved to {REPORT_PATH}")
    print(f"[EVAL] Confusion matrices saved to {CONFUSION_TRAIN_IMG} and {CONFUSION_TEST_IMG}")
    print(f"[EVAL] Precision bar chart saved to {PRECISION_IMG}")


def _plot_confusion(cm: np.ndarray, labels: List[str], out_path: Path, title: str) -> None:
    plt.figure(figsize=(12, 10))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité terrain")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _plot_precision(train_prec: float, test_prec: float, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Train precision", "Test precision"], [train_prec, test_prec], color=["#4C78A8", "#F58518"])
    plt.ylim(0, 1)
    for bar, val in zip(bars, [train_prec, test_prec]):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", va="bottom")
    plt.ylabel("Macro precision")
    plt.title("Precision comparison (train vs test)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ================= MAIN =================
def main(force_recompute: bool = False):
    from train import load_artifacts

    train_df, test_df, binarizer, _ = load_artifacts()
    classes = list(binarizer.classes_)

    train_feat_path = build_feature_store(train_df, "train", force=force_recompute)
    test_feat_path = build_feature_store(test_df, "test", force=force_recompute)

    scaler = fit_scaler(train_feat_path, len(train_df))
    train_scaled = transform_split(train_feat_path, len(train_df), scaler, "train")
    test_scaled = transform_split(test_feat_path, len(test_df), scaler, "test")

    knn = train_knn(train_scaled, train_df["label"].values)
    evaluate(knn, train_scaled, train_df["label"].values, test_scaled, test_df["label"].values, classes)


if __name__ == "__main__":
    main()
