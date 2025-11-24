import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence, to_categorical

# ====================================================================
#                      CONSTANTS AND CONFIG
# ====================================================================
BS = 64  # 32 if you want more augmentation per sample, 64 if memory allows
WIDTH = 256
HEIGHT = 256
DEPTH = 3
DEFAULT_IMAGE_SIZE = (WIDTH, HEIGHT)
INIT_LR = 1e-3
EPOCHS = 25
BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "artifacts"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MAX_ROT_DEG = 15
COLOR_JITTER = 0.08  # +/- 8% on brightness/saturation


# ====================================================================
#           GENERATOR CLASS (RAM-friendly)
# ====================================================================
class DataGenerator(Sequence):
    """Keras data generator to load images in batches."""

    def __init__(self, df, labels_encoded, batch_size=BS, dim=DEFAULT_IMAGE_SIZE, n_channels=3,
                 shuffle=True, augment=False):
        self.df = df.reset_index(drop=True)
        self.labels_encoded = np.array(labels_encoded)
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        # Use ceil to avoid losing samples on the last batch.
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        list_paths_temp = self.df['path'].iloc[indices].tolist()
        labels_temp = self.labels_encoded[indices]
        X, y = self.__data_generation(list_paths_temp, labels_temp)
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, list_paths_temp, labels_temp):
        batch_size = len(list_paths_temp)
        X = np.empty((batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = labels_temp

        for i, path in enumerate(list_paths_temp):
            image = cv2.imread(path)
            if image is None:
                raise RuntimeError(f"Could not load image: {path}")
            image = cv2.resize(image, self.dim)
            if self.augment:
                image = self._augment(image)
            X[i,] = image.astype('float32') / 255.0

        return X, y

    def _augment(self, img):
        # Horizontal flip with probability 0.5
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)

        # Small random rotation
        angle = np.random.uniform(-MAX_ROT_DEG, MAX_ROT_DEG)
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT101)

        # Light color jitter (brightness/saturation)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 2] *= np.random.uniform(1 - COLOR_JITTER, 1 + COLOR_JITTER)
        hsv[..., 1] *= np.random.uniform(1 - COLOR_JITTER, 1 + COLOR_JITTER)
        hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img


def load_artifacts():
    print("## 1. 🔄 Loading data artifacts")
    train_csv = SAVE_DIR / 'train_df.csv'
    test_csv = SAVE_DIR / 'test_df.csv'
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"CSV files not found in {SAVE_DIR}. Run load_files.py first.")

    loaded_train_df = pd.read_csv(train_csv)
    loaded_test_df = pd.read_csv(test_csv)
    print("✅ Train and test DataFrames loaded.")

    binarizer_path = SAVE_DIR / 'label_binarizer.pkl'
    if not binarizer_path.exists():
        raise FileNotFoundError(f"{binarizer_path} missing. Run load_files.py first.")
    with open(binarizer_path, 'rb') as f:
        loaded_binarizer: LabelBinarizer = pickle.load(f)
    n_classes = len(loaded_binarizer.classes_)
    print(f"✅ LabelBinarizer loaded. {n_classes} classes detected.")

    return loaded_train_df, loaded_test_df, loaded_binarizer, n_classes


def encode_labels(train_df, test_df, binarizer: LabelBinarizer, n_classes: int):
    """Encode labels to one-hot using the binarizer class order."""
    label_map = {name: i for i, name in enumerate(binarizer.classes_)}
    train_indices = train_df['label'].map(label_map).values
    test_indices = test_df['label'].map(label_map).values

    train_labels_encoded = to_categorical(train_indices, num_classes=n_classes)
    test_labels_encoded = to_categorical(test_indices, num_classes=n_classes)

    return train_labels_encoded, test_labels_encoded, train_indices, test_indices


def log_class_stats(train_df, class_weights, classes):
    """Print image distribution and class weights."""
    counts = train_df['label'].value_counts().sort_index()
    print("\n[INFO] Final train distribution and class weights:")
    for idx, cls in enumerate(classes):
        count = counts.get(cls, 0)
        weight = class_weights.get(idx, 0)
        print(f"  - {cls:40s} : {count:5d} images | weight = {weight:.3f}")
    total = counts.sum()
    print(f"  Total images (train) : {total}")


def plot_history(history_dict, out_dir: Path):
    """Save loss/accuracy plots."""
    acc_key = next((k for k in history_dict.keys() if "acc" in k and not k.startswith("val_")), None)
    val_acc_key = next((k for k in history_dict.keys() if "val" in k and "acc" in k), None)
    if acc_key is None or val_acc_key is None:
        return None

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], label='train_loss')
    plt.plot(history_dict['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_dict[acc_key], label='train_acc')
    plt.plot(history_dict[val_acc_key], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    out_path = out_dir / "training_history.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def save_confusion_matrix(y_true, y_pred, classes, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_metrics_text(y_true, y_pred, classes, out_path: Path):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    with open(out_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)


def build_model(n_classes: int):
    inputShape = (HEIGHT, WIDTH, DEPTH)
    chanDim = -1
    if K.image_data_format() == "channels_first":
        inputShape = (DEPTH, HEIGHT, WIDTH)
        chanDim = 1

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(252))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))

    model.compile(optimizer=Adam(learning_rate=INIT_LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    loaded_train_df, loaded_test_df, loaded_binarizer, n_classes = load_artifacts()

    print("\n## 2. Label preparation and cleaning")
    train_labels_encoded, test_labels_encoded, train_indices, test_indices = encode_labels(
        loaded_train_df, loaded_test_df, loaded_binarizer, n_classes
    )

    # Class weights to mitigate imbalance (accounts for rare classes)
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=train_indices
    )
    class_weights = {i: w for i, w in enumerate(weights)}
    log_class_stats(loaded_train_df, class_weights, loaded_binarizer.classes_)

    # Build Keras generators
    train_generator = DataGenerator(
        loaded_train_df, train_labels_encoded, batch_size=BS, shuffle=True, augment=True
    )
    test_generator = DataGenerator(
        loaded_test_df, test_labels_encoded, batch_size=BS, shuffle=False, augment=False
    )
    if len(train_generator) == 0 or len(test_generator) == 0:
        raise RuntimeError("Generators are empty. Check data preparation.")

    print(f"[INFO] Train samples: {len(loaded_train_df)}")
    print(f"[INFO] Test samples: {len(loaded_test_df)}")

    # ====================================================================
    #                      MODEL DEFINITION AND TRAINING
    # ====================================================================
    print("\n## 3. Model definition and training 🧠")
    model = build_model(n_classes)
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data=test_generator,
        validation_steps=len(test_generator),
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # ====================================================================
    #                      EVALUATION AND VISUALS
    # ====================================================================
    history_path = plot_history(history.history, SAVE_DIR)

    print("\n[INFO] Predicting on test set...")
    preds = model.predict(test_generator, verbose=1)
    y_true = np.array(test_indices)
    y_pred = np.argmax(preds, axis=1)

    cm_path = SAVE_DIR / "confusion_matrix.png"
    save_confusion_matrix(y_true, y_pred, loaded_binarizer.classes_, cm_path)

    metrics_path = SAVE_DIR / "metrics.txt"
    save_metrics_text(y_true, y_pred, loaded_binarizer.classes_, metrics_path)

    if history_path:
        print(f"📈 Training/validation curves saved to: {history_path}")
    print(f"📊 Confusion matrix saved to: {cm_path}")
    print(f"📝 Metrics (accuracy, precision/recall/f1) saved to: {metrics_path}")

    print("\n[FIN] Training complete!")

    # ====================================================================
    #                      MODEL SAVE
    # ====================================================================
    model_path = SAVE_DIR / 'plant_disease_model.keras'
    model.save(model_path)
    print(f"✅ Keras model saved to: {model_path}")


if __name__ == "__main__":
    # Enable GPU memory growth if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except Exception:
            pass
    main()
