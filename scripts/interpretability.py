import argparse
import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "plant_disease_model.keras"
BINARIZER_PATH = ARTIFACTS_DIR / "label_binarizer.pkl"
TRAIN_CSV = ARTIFACTS_DIR / "train_df.csv"
TEST_CSV = ARTIFACTS_DIR / "test_df.csv"
IMAGE_SIZE: Tuple[int, int] = (256, 256)
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "interpretability"


def load_model_and_labels():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
    if not BINARIZER_PATH.exists():
        raise FileNotFoundError(f"Binarizer not found at {BINARIZER_PATH}. Run load_files.py first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(BINARIZER_PATH, "rb") as f:
        binarizer = pickle.load(f)
    return model, binarizer


def preprocess_image(img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return original BGR image and normalized batch tensor."""
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Could not read image: {img_path}")
    resized = cv2.resize(img_bgr, IMAGE_SIZE)
    tensor = resized.astype("float32") / 255.0
    return img_bgr, np.expand_dims(tensor, axis=0)


def ensure_model_built(model: tf.keras.Model):
    """Ensure the model is built so .inputs/.outputs are available."""
    try:
        _ = model.outputs  # noqa: F841
        return
    except Exception:
        pass

    symbolic_input = tf.keras.Input(shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3))
    _ = model(symbolic_input)
    return


def find_last_conv_layer(model: tf.keras.Model):
    """Return (layer, index) for the last conv-like layer for Grad-CAM."""
    for idx in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[idx]
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer, idx

        out_shape = getattr(layer, "output_shape", None)
        if out_shape is None:
            continue

        try:
            if len(out_shape) == 4:
                return layer, idx
        except TypeError:
            try:
                if len(out_shape[0]) == 4:
                    return layer, idx
            except Exception:
                continue

    raise RuntimeError("No convolutional layer found for Grad-CAM. Ensure the model has Conv2D layers.")


def compute_gradcam_heatmap(model: tf.keras.Model, img_tensor: np.ndarray, class_idx: int | None):
    ensure_model_built(model)
    last_conv, conv_idx = find_last_conv_layer(model)

    # Split model: feature extractor up to last conv, classifier head after it.
    model_inputs = model.inputs
    feature_extractor = tf.keras.Model(inputs=model_inputs, outputs=last_conv.output)
    classifier_input = tf.keras.Input(shape=last_conv.output.shape[1:])
    x = classifier_input
    for layer in model.layers[conv_idx + 1:]:
        x = layer(x)
    classifier = tf.keras.Model(inputs=classifier_input, outputs=x)

    img_tensor = tf.cast(img_tensor, tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(img_tensor, training=False)
        tape.watch(conv_outputs)
        preds = classifier(conv_outputs, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        class_channel = preds[:, class_idx]
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None for Grad-CAM. Check that the model is differentiable and inputs are float32.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(class_idx)


def overlay_heatmap(heatmap: np.ndarray, original_bgr: np.ndarray) -> np.ndarray:
    heatmap_resized = cv2.resize(heatmap, (original_bgr.shape[1], original_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 0.55, heatmap_color, 0.45, 0)
    return overlay


def run_gradcam(model: tf.keras.Model, binarizer, image_paths: Sequence[Path], output_dir: Path):
    for img_path in image_paths:
        original_bgr, tensor = preprocess_image(img_path)
        heatmap, class_idx = compute_gradcam_heatmap(model, tensor, class_idx=None)
        overlay = overlay_heatmap(heatmap, original_bgr)
        label = binarizer.classes_[class_idx]
        cv2.putText(overlay, f"Grad-CAM: {label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        out_path = output_dir / f"{img_path.stem}_gradcam.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"Grad-CAM saved to {out_path}")


def run_lime(model: tf.keras.Model, binarizer, image_paths: Sequence[Path], output_dir: Path, num_samples: int):
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(batch):
        resized_batch = []
        for img in batch:
            img_uint8 = np.clip(img, 0, 255).astype("uint8")
            resized = cv2.resize(img_uint8, IMAGE_SIZE)
            resized_batch.append(resized.astype("float32") / 255.0)
        preds = model.predict(np.array(resized_batch), verbose=0)
        return preds

    for img_path in image_paths:
        original_bgr, _ = preprocess_image(img_path)
        explanation = explainer.explain_instance(
            np.copy(original_bgr),
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )
        label_idx = explanation.top_labels[0]
        label = binarizer.classes_[label_idx]
        temp, mask = explanation.get_image_and_mask(
            label=label_idx,
            positive_only=False,
            num_features=8,
            hide_rest=False
        )
        boundaries = mark_boundaries(temp.astype(np.float32) / 255.0, mask)
        lime_rgb = (boundaries * 255).astype(np.uint8)
        lime_bgr = cv2.cvtColor(lime_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(lime_bgr, f"LIME: {label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        out_path = output_dir / f"{img_path.stem}_lime.png"
        cv2.imwrite(str(out_path), lime_bgr)
        print(f"LIME explanation saved to {out_path}")


def load_background_samples(sample_size: int) -> np.ndarray:
    source_csv = TRAIN_CSV if TRAIN_CSV.exists() else TEST_CSV
    if not source_csv.exists():
        raise FileNotFoundError("No train/test CSV found for SHAP background. Run load_files.py first.")

    df = pd.read_csv(source_csv)
    sampled = df.sample(min(sample_size, len(df)), random_state=42)
    background = []
    for img_path in sampled["path"].tolist():
        _, tensor = preprocess_image(Path(img_path))
        background.append(tensor[0])
    return np.array(background)


def run_shap(model: tf.keras.Model, binarizer, image_paths: Sequence[Path], output_dir: Path, background_size: int):
    if background_size <= 0:
        raise ValueError("background_size must be positive for SHAP.")

    with tf.device("/CPU:0"):
        background = load_background_samples(background_size)
        explainer = shap.DeepExplainer(model, background)

    originals: List[np.ndarray] = []
    batch = []
    for img_path in image_paths:
        original_bgr, tensor = preprocess_image(img_path)
        originals.append(original_bgr)
        batch.append(tensor[0])

    batch_arr = np.array(batch)
    preds = model.predict(batch_arr, verbose=0)

    for i, img_path in enumerate(image_paths):
        class_idx = int(np.argmax(preds[i]))
        label = binarizer.classes_[class_idx]
        try:
            with tf.device("/CPU:0"):
                shap_values = explainer.shap_values(batch_arr[i:i + 1], check_additivity=False)
        except tf.errors.ResourceExhaustedError as e:
            raise RuntimeError(
                "SHAP ran out of memory. Try lowering --background-size, explaining fewer images, "
                "or skipping SHAP for this run."
            ) from e

        # Handle different SHAP return shapes (list per class vs single array)
        if isinstance(shap_values, list):
            if len(shap_values) == 1:
                shap_map = shap_values[0][0]
            elif class_idx < len(shap_values):
                shap_map = shap_values[class_idx][0]
            else:
                shap_map = shap_values[-1][0]
                print(f"[WARN] SHAP returned {len(shap_values)} outputs; using last index instead of class {class_idx}.")
        else:
            shap_map = shap_values[0]

        heatmap = np.mean(shap_map, axis=-1)
        heatmap = np.abs(heatmap)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        overlay = overlay_heatmap(heatmap, originals[i])
        cv2.putText(overlay, f"SHAP: {label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        out_path = output_dir / f"{img_path.stem}_shap.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"SHAP attribution saved to {out_path}")


def load_test_samples(sample_size: int) -> List[Path]:
    if sample_size <= 0:
        return []
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"{TEST_CSV} not found. Run load_files.py to generate it.")
    df = pd.read_csv(TEST_CSV)
    if df.empty:
        raise RuntimeError("Test CSV is empty; cannot sample images.")
    sample_size = min(sample_size, len(df))
    sampled = df.sample(sample_size, random_state=None)
    return [Path(p) for p in sampled["path"].tolist()]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM, LIME, and SHAP explanations for images.")
    parser.add_argument("images", nargs="*", help="Path(s) to images to explain. Optional if --sample-test is set.")
    parser.add_argument("--methods", nargs="+", default=["gradcam", "lime", "shap"],
                        choices=["gradcam", "lime", "shap"], help="Which interpretability methods to run.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory to store explanation images.")
    parser.add_argument("--background-size", type=int, default=20,
                        help="Number of validation images to use as SHAP background (smaller uses less memory).")
    parser.add_argument("--lime-samples", type=int, default=1000,
                        help="Number of perturbations for LIME.")
    parser.add_argument("--sample-test", type=int, default=0,
                        help="If >0, randomly pick this many images from the test split.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_paths = [Path(p) for p in args.images]

    if args.sample_test > 0:
        sampled = load_test_samples(args.sample_test)
        image_paths.extend(sampled)

    if not image_paths:
        raise ValueError("No images provided. Either pass paths or use --sample-test to auto-sample from test_df.csv.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, binarizer = load_model_and_labels()
    methods = set(args.methods)

    if "gradcam" in methods:
        run_gradcam(model, binarizer, image_paths, output_dir)
    if "lime" in methods:
        run_lime(model, binarizer, image_paths, output_dir, num_samples=args.lime_samples)
    if "shap" in methods:
        run_shap(model, binarizer, image_paths, output_dir, background_size=args.background_size)


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except Exception:
            pass
    main()
