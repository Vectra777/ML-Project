import numpy as np
import cv2
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ================= CONFIG =================
BLOCK_SIZE = 300
N_NEIGHBORS = 3
IMAGE_SIZE = (64, 64)
CNN_SIZE = (224, 224)
cnn_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
CNN_FEATURES = int(np.prod(cnn_model.output_shape[1:]))
FEATURE_DIM = IMAGE_SIZE[0] * IMAGE_SIZE[1] + CNN_FEATURES

# ================= FEATURE EXTRACTION =================
def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Impossible de lire l'image : {img_path}")
    
    gray = cv2.cvtColor(cv2.resize(img, IMAGE_SIZE), cv2.COLOR_BGR2GRAY)
    pixels = gray.flatten()

    img_cnn = cv2.resize(img, CNN_SIZE)
    img_cnn = img_to_array(img_cnn)
    img_cnn = np.expand_dims(img_cnn, axis=0)
    img_cnn = preprocess_input(img_cnn)
    cnn_feat = cnn_model.predict(img_cnn, verbose=0).flatten()

    return np.hstack([pixels, cnn_feat])

# ================= KNN GENERATOR =================
class KNNDataGenerator:
    def __init__(self, df, block_size=BLOCK_SIZE, shuffle=True, seed=42):
        self.df = df.reset_index(drop=True)
        self.block_size = block_size
        self.n_samples = len(df)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        # stratify by class
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(df['label']):
            self.class_indices[label].append(idx)

        if self.shuffle:
            for k, v in self.class_indices.items():
                self.rng.shuffle(v)

        self.remaining_indices = {k: v.copy() for k, v in self.class_indices.items()}
        self.class_ratios = {k: len(v) / self.n_samples for k, v in self.class_indices.items()}

    def __iter__(self):
        return self

    def __next__(self):
        if sum(len(v) for v in self.remaining_indices.values()) == 0:
            raise StopIteration

        block_indices = []
        for label, ratio in self.class_ratios.items():
            n = int(ratio*self.block_size)
            take = min(n, len(self.remaining_indices[label]))
            block_indices.extend(self.remaining_indices[label][:take])
            self.remaining_indices[label] = self.remaining_indices[label][take:]

        # compléter si bloc trop petit
        while len(block_indices) < self.block_size and sum(len(v) for v in self.remaining_indices.values()) > 0:
            for label, lst in self.remaining_indices.items():
                if lst:
                    block_indices.append(lst.pop(0))
                    if len(block_indices) >= self.block_size:
                        break

        if not block_indices:
            raise StopIteration

        block = self.df.iloc[block_indices]
        X_block = np.array([extract_features(p) for p in block['path']])
        y_block = block['label'].values
        return X_block, y_block

# ================= TRAIN KNN PAR BLOC =================
def train_knn_blocks(df, block_size=BLOCK_SIZE, n_neighbors=N_NEIGHBORS):
    generator = KNNDataGenerator(df, block_size=block_size)
    knns = []
    scalers = []
    block_sizes = []
    print(f"[TRAIN] Samples: {len(df)} | Classes: {df['label'].nunique()} | Block size: {block_size}")
    print(f"[TRAIN] Feature dim: {FEATURE_DIM} (pixels {IMAGE_SIZE} + VGG16 conv features)")
    for X_block, y_block in generator:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_block)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_scaled, y_block)
        knns.append(knn)
        scalers.append(scaler)
        block_sizes.append(len(y_block))
        top_class = None
        if len(y_block):
            uniq, counts = np.unique(y_block, return_counts=True)
            top_class = uniq[np.argmax(counts)]
        print(f"[TRAIN] Bloc de {len(y_block)} images entraîné. Classe la plus fréquente: {top_class}")
    print(f"[TRAIN] Total blocs entraînés: {len(knns)}")
    return knns, scalers, block_sizes

# ================= PREDICTION FINALE =================
def predict_knn_blocks(knns, scalers, block_sizes, df):
    y_preds = []
    total = len(df)
    print(f"[PRED] Démarrage des prédictions sur {total} images (ensemble de {len(knns)} KNN)")
    for idx, row in df.iterrows():
        feats = extract_features(row['path'])
        # vote pondéré par la taille du bloc (plus un bloc a d'échantillons, plus son vote compte)
        vote_weights = defaultdict(float)
        for knn, scaler, size in zip(knns, scalers, block_sizes):
            pred = knn.predict(scaler.transform([feats]))[0]
            vote_weights[pred] += size
        final_pred = max(vote_weights.items(), key=lambda kv: kv[1])[0]
        y_preds.append(final_pred)
        if (idx + 1) % 200 == 0 or idx + 1 == total:
            print(f"[PRED] {idx + 1}/{total} images traitées")
    return np.array(y_preds)

# ================= UTILISATION =================
if __name__ == "__main__":
    # Reuse the existing artifact loader from the CNN pipeline
    from train import load_artifacts

    loaded_train_df, loaded_test_df, loaded_binarizer, n_classes = load_artifacts()

    # Entraîner KNN par bloc stratifié
    knns, scalers, block_sizes = train_knn_blocks(loaded_train_df)

    # Prédire sur le test set
    y_pred = predict_knn_blocks(knns, scalers, block_sizes, loaded_test_df)

    # Évaluation
    y_true = loaded_test_df['label'].values
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy ensemble KNN : {acc:.4f}")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix shape:", cm.shape)
