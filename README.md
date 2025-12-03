# Plant Disease Classification

CNN-based classifier for PlantVillage leaf images. Includes utilities to fetch/clean the Kaggle dataset, train a Keras model, and run predictions on new photos.

## Project Structure
- `scripts/load_files.py`: downloads the Kaggle dataset, consolidates folders, builds `train_df.csv` and `test_df.csv`, and stores a `LabelBinarizer` in `scripts/artifacts/`.
- `scripts/train.py`: trains a custom ConvNet (TensorFlow/Keras), plots training curves, exports `plant_disease_model.keras`, `confusion_matrix.png`, and `metrics.txt`.
- `scripts/utilisation.py`: CLI inference tool to predict the top classes for a leaf image.
- `scripts/interpretability.py`: generates Grad-CAM, LIME, and SHAP visualizations for sample images.
- `scripts/artifacts/`: cached artifacts (model, metrics, plots, train/test CSVs). Safe to delete/regenerate.
- `scripts/plantdisease_working/PlantVillage`: writable copy of the dataset (created by `load_files.py`).
- `notebooks/`: exploratory work and usage demos.

## Setup
1) Install Python 3.10+ and (optionally) create a virtual env:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```
2) Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3) Kaggle access (needed for `load_files.py`): place your `kaggle.json` in `~/.kaggle/` or set `KAGGLE_USERNAME` and `KAGGLE_KEY`.
4) GPU is optional but recommended; TensorFlow will use CUDA if available.

## Prepare Data
Download and stage the PlantVillage dataset (via KaggleHub), then create train/test splits and label mappings:
```bash
python scripts/load_files.py
```
Outputs:
- `scripts/plantdisease_working/PlantVillage/` (writable dataset copy)
- `scripts/artifacts/train_df.csv`, `test_df.csv`
- `scripts/artifacts/label_binarizer.pkl`

If the dataset is already present in `scripts/plantdisease_working/PlantVillage`, the script will reuse it.

## Train the Model
```bash
python scripts/train.py
```
Key parameters (defined at the top of `train.py`): image size 256×256, batch size 64, learning rate 1e-3, 25 epochs, light augmentation and class weighting. Artifacts saved to `scripts/artifacts/`:
- `plant_disease_model.keras`
- `training_history.png`
- `confusion_matrix.png`
- `metrics.txt` (includes accuracy and per-class precision/recall/F1; current run ≈98% accuracy on the held-out split)

## Inference
Classify a single image (requires the saved model and binarizer):
```bash
python scripts/utilisation.py path/to/leaf.jpg --top 3
```
Example output:
```
Best prediction: Tomato_Leaf_Mold (prob=0.9835)
Other predictions:
  - Tomato_Bacterial_spot (prob=0.0124)
  - Tomato__Target_Spot (prob=0.0031)
```

## Interpretability (Grad-CAM, LIME, SHAP)
Generate attribution overlays using the trained model and test split:
```bash
# Auto-sample 2 test images and run all methods
python scripts/interpretability.py --sample-test 2

# Provide specific images and lower SHAP background for memory
python scripts/interpretability.py img1.jpg img2.jpg --background-size 10

# Only run Grad-CAM and LIME
python scripts/interpretability.py img1.jpg --methods gradcam lime
```
Outputs are saved to `scripts/artifacts/interpretability/`. SHAP can be memory-heavy; reduce `--background-size` or skip SHAP if you hit OOM.

## Notebooks
- `notebooks/projet_ai_V1.ipynb` / `projet_ai_V2.ipynb`: data exploration and model iterations.
- `notebooks/usage.ipynb`: example inference workflow.
- `notebooks/tests.ipynb`: quick checks and utility experiments.

## Tips
- If you rerun `load_files.py`, artifacts and the working dataset are refreshed.
- To start clean, delete `scripts/artifacts/` and `scripts/plantdisease_working/` before running the pipeline.
- For faster training, use a GPU-enabled TensorFlow build (`tensorflow[and-cuda]` in `requirements.txt`).
