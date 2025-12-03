import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Paths and constants
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "plant_disease_model.keras"
BINARIZER_PATH = ARTIFACTS_DIR / "label_binarizer.pkl"
IMAGE_SIZE = (256, 256)


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
    if not BINARIZER_PATH.exists():
        raise FileNotFoundError(f"Binarizer not found at {BINARIZER_PATH}.")

    model = load_model(MODEL_PATH)
    with open(BINARIZER_PATH, "rb") as f:
        binarizer = pickle.load(f)
    return model, binarizer


def preprocess_image(img_path: Path):
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Could not read image: {img_path}")
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


def predict_image(img_path: Path, top_k: int = 5):
    model, binarizer = load_artifacts()
    input_tensor = preprocess_image(img_path)

    preds = model.predict(input_tensor, verbose=0)[0]
    indices = np.argsort(preds)[::-1][:top_k]

    results = []
    for idx in indices:
        cls = binarizer.classes_[idx]
        prob = float(preds[idx])
        results.append((cls, prob))
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict the class of a leaf image.")
    parser.add_argument("image_path", type=str, help="Path to the image to classify.")
    parser.add_argument("--top", type=int, default=5, help="Number of predictions to return (max).")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    top_k = max(1, args.top)

    results = predict_image(img_path, top_k=top_k)
    best = results[0]
    others = results[1:5]

    print(f"\nBest prediction: {best[0]} (prob={best[1]:.4f})")
    if others:
        print("Other predictions:")
        for cls, prob in others:
            print(f"  - {cls} (prob={prob:.4f})")


if __name__ == "__main__":
    # Prepare GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except Exception:
            pass
    main()
