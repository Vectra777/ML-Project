from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
from pathlib import Path
import pickle
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Initialize FastAPI
app = FastAPI(title="Image Classification API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_DIR = Path("models")
BINARIZER_PATH = "label_transform.pkl"
SKLEARN_MODEL_CONFIGS = {
    "rf_model": {"model": MODEL_DIR / "rf_model.pkl", "scaler": MODEL_DIR / "rf_scaler.pkl"},
    "knn_model": {"model": MODEL_DIR / "knn_model.pkl", "scaler": MODEL_DIR / "knn_scaler.pkl"},
}
FEATURE_EXTRACTOR_SIZE = (224, 224)
FRONTEND_INDEX = Path("index.html")

# Load shared label binarizer
def load_label_binarizer():
    for path in (MODEL_DIR / "label_binarizer.pkl", Path(BINARIZER_PATH)):
        if path.exists():
            with open(path, "rb") as f:
                lb = pickle.load(f)
            print(f"✔ Loaded shared binarizer from {path} with {len(lb.classes_)} classes.")
            return lb
    raise FileNotFoundError("No label binarizer found (expected models/label_binarizer.pkl or label_transform.pkl).")


label_binarizer = load_label_binarizer()

CLASSES = list(label_binarizer.classes_)

# Load all models
keras_models = {}
sklearn_models = {}
model_input_shapes = {}

print("🔍 Loading CNN (.keras) models...")
for model_path in MODEL_DIR.glob("*.keras"):
    name = model_path.stem
    print(f"→ Loading {name}...")
    model = tf.keras.models.load_model(model_path, compile=False)
    keras_models[name] = model

    shape = model.input_shape
    if shape and shape[1] and shape[2]:
        model_input_shapes[name] = (int(shape[1]), int(shape[2]))
    else:
        model_input_shapes[name] = (256, 256)

# Feature extractor for classical ML models
print("🔍 Loading MobileNetV2 feature extractor for classical models...")
feature_extractor = MobileNetV2(
    weights="imagenet", include_top=False, pooling="avg", input_shape=(*FEATURE_EXTRACTOR_SIZE, 3)
)
feature_vector_size = int(np.prod(feature_extractor.output_shape[1:]))

print("🔍 Loading classical (.pkl) models...")
for name, paths in SKLEARN_MODEL_CONFIGS.items():
    model_path, scaler_path = paths["model"], paths["scaler"]
    if not model_path.exists() or not scaler_path.exists():
        print(f"⚠️ Skipping {name}: expected {model_path.name} and {scaler_path.name} in {MODEL_DIR}")
        continue
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    sklearn_models[name] = {"model": model, "scaler": scaler}
    print(f"→ Loaded {name} (feature dim={feature_vector_size})")

available_models = list(keras_models.keys()) + list(sklearn_models.keys())
print("✔ All models loaded.")
print("Available models:", available_models)


def prepare_cnn_input(pil_image: Image.Image, target_size):
    """Resize/normalize image for Keras CNNs."""
    img = np.array(pil_image)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def extract_features(pil_image: Image.Image):
    """Extract pooled CNN features for classical ML models."""
    img = np.array(pil_image)
    img = cv2.resize(img, FEATURE_EXTRACTOR_SIZE)
    img = img.astype("float32")
    img = preprocess_input(img)
    batch = np.expand_dims(img, axis=0)
    feats = feature_extractor.predict(batch, verbose=0)
    if feats.ndim > 2:
        feats = feats.reshape((feats.shape[0], -1))
    return feats


# Endpoints
@app.get("/models")
async def list_models():
    return {"available_models": available_models}


@app.get("/")
async def serve_index():
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    return JSONResponse({"detail": "index.html not found"}, status_code=404)


@app.post("/predict")
async def predict(
    image: UploadFile,
    model: str = Form(...)
):
    if model not in available_models:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid or missing model",
                "available": available_models
            }
        )

    # Read image once
    img_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # CNN models
    if model in keras_models:
        target_h, target_w = model_input_shapes[model]
        batch = prepare_cnn_input(pil_image, (target_w, target_h))
        probs = keras_models[model].predict(batch, verbose=0)
        probs = np.asarray(probs)

        # Softmax if needed
        if not np.all((probs >= 0) & (probs <= 1)):
            exp = np.exp(probs - np.max(probs))
            probs = exp / np.sum(exp)

    # Classical models (RF / KNN)
    else:
        feats = extract_features(pil_image)
        sklearn_bundle = sklearn_models[model]
        scaled = sklearn_bundle["scaler"].transform(feats)
        if hasattr(sklearn_bundle["model"], "predict_proba"):
            probs = sklearn_bundle["model"].predict_proba(scaled)
        else:
            # Fallback: use decision function then softmax
            logits = sklearn_bundle["model"].decision_function(scaled)
            exp = np.exp(logits - np.max(logits))
            probs = exp / np.sum(exp, axis=1, keepdims=True)

    top_idx = int(np.argmax(probs))
    top_prob = float(probs[0][top_idx])
    pred_label = CLASSES[top_idx]

    return JSONResponse({
        "model": model,
        "prediction": pred_label,
        "confidence": round(top_prob, 4),
        "class_index": top_idx,
        "probabilities": {CLASSES[i]: float(probs[0][i]) for i in range(len(probs[0]))}
    })
