from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
from pathlib import Path
import pickle

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

# Load shared label binarizer
with open(BINARIZER_PATH, "rb") as f:
    label_binarizer = pickle.load(f)

CLASSES = list(label_binarizer.classes_)
print(f"✔ Loaded shared binarizer with {len(CLASSES)} classes.")

# Load all models
models = {}
model_input_shapes = {}

print("🔍 Loading models...")
for model_path in MODEL_DIR.glob("*.keras"):
    name = model_path.stem
    print(f"→ Loading {name}...")
    model = tf.keras.models.load_model(model_path, compile=False)
    models[name] = model

    shape = model.input_shape
    if shape and shape[1] and shape[2]:
        model_input_shapes[name] = (int(shape[1]), int(shape[2]))
    else:
        model_input_shapes[name] = (256, 256)

print("✔ All models loaded.")
print("Available models:", list(models.keys()))


# Endpoints
@app.get("/models")
async def list_models():
    return {"available_models": list(models.keys())}


@app.post("/predict")
async def predict(
    image: UploadFile,
    model: str = Form(...)
):
    if model not in models:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid or missing model",
                "available": list(models.keys())
            }
        )

    model_instance = models[model]
    target_h, target_w = model_input_shapes[model]

    # Read and process image
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (target_w, target_h))
    img = img.astype("float32") / 255.0
    batch = np.expand_dims(img, axis=0)

    # Predict
    probs = model_instance.predict(batch)
    probs = np.asarray(probs)

    # Softmax if needed
    if not np.all((probs >= 0) & (probs <= 1)):
        exp = np.exp(probs - np.max(probs))
        probs = exp / np.sum(exp)

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
