import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# ====================================================================
#                      CONFIGURATION AND CONSTANTS
# ====================================================================
DATASET_REF        = "emmarex/plantdisease"
WORKING_DIR_NAME   = "plantdisease_working"  # Writable copy of the dataset
ALLOWED_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
MAX_PER_CLASS      = 100000  # Cap images per class
ARTIFACTS_DIR      = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ====================================================================
#                           UTILITY FUNCTIONS
# ====================================================================
def _find_class_dirs(root: Path) -> List[Path]:
    """Find every folder containing images and treat it as a class."""
    class_dirs = []
    for p in root.iterdir():
        if p.is_dir() and any(f.is_file() and f.suffix.lower() in ALLOWED_EXTS for f in p.iterdir()):
            class_dirs.append(p)
    return sorted(class_dirs, key=lambda x: x.name)


def consolidate_dataset(read_only_path: str) -> str:
    """
    Copy the dataset to a writable folder and collapse duplicate paths.
    Needed to avoid "Read-only file system" errors.
    """
    print(f"\n## 🔄 2. Copy to writable folder and clean")

    # Define and create the working path (writable)
    working_base_dir = os.path.join(os.getcwd(), WORKING_DIR_NAME)
    if os.path.exists(working_base_dir):
        shutil.rmtree(working_base_dir)

    # Copy the full dataset
    print(f"   -> Copying {read_only_path} to {working_base_dir}...")
    shutil.copytree(read_only_path, working_base_dir)
    base_dir = working_base_dir

    # Remove duplicated folder paths (ex: plantdisease_working/plantvillage/PlantVillage)
    target_root_name = "PlantVillage"
    target_root = os.path.join(base_dir, target_root_name)
    duplicated_root = os.path.join(base_dir, "plantvillage", target_root_name)

    if os.path.exists(duplicated_root):
        print(f"💡 Found duplicated folders. Consolidating...")
        try:
            for class_name in os.listdir(duplicated_root):
                source_folder = os.path.join(duplicated_root, class_name)
                target_folder = os.path.join(target_root, class_name)

                if os.path.isdir(source_folder):
                    # Ensure target folder exists
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)

                    # Move files now that we are in a writable location
                    for item_name in os.listdir(source_folder):
                        source_item = os.path.join(source_folder, item_name)
                        target_item = os.path.join(target_folder, item_name)

                        if os.path.isfile(source_item) and item_name.endswith(tuple(ALLOWED_EXTS)):
                            shutil.move(source_item, target_item)

            # Remove empty duplicated structure
            shutil.rmtree(os.path.join(base_dir, "plantvillage"))
            print("✅ Consolidation done.")

        except Exception as e:
            print(f"❌ Error while moving files: {e}")

    return os.path.join(base_dir, 'PlantVillage')  # Return the class root


def plot_class_distribution_global(class_counts: dict):
    """Display class distribution before the split."""
    if not class_counts:
        return
    labels = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.style.use('ggplot')
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x=labels, y=counts, palette="viridis")
    ax.set_title(f"Images per class (capped at {MAX_PER_CLASS})", fontsize=16)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Image count", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    for index, value in enumerate(counts):
        plt.text(index, value + (max(counts) * 0.01), str(value), ha='center')
    plt.tight_layout()
    plt.show()


def print_class_distribution(labels: np.ndarray, name: str = "Dataset") -> Tuple[np.ndarray, np.ndarray]:
    """Print class distribution in percentages."""
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nClass distribution in '{name}':")
    for cls, count in zip(unique, counts):
        print(f"  {cls} : {count} ({count / len(labels) * 100:.2f}%)")
    return unique, counts


def build_dataframe(dataset_root: Path) -> Tuple[pd.DataFrame, dict]:
    """Walk the class tree and build a path/label DataFrame."""
    records = []
    class_counts = defaultdict(int)

    for class_dir in _find_class_dirs(dataset_root):
        for img_path in class_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in ALLOWED_EXTS:
                if class_counts[class_dir.name] >= MAX_PER_CLASS:
                    continue
                records.append({"path": str(img_path.resolve()), "label": class_dir.name})
                class_counts[class_dir.name] += 1

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No valid images found in the consolidated dataset.")

    return df, class_counts


def save_artifacts(train_df: pd.DataFrame, test_df: pd.DataFrame, binarizer: LabelBinarizer) -> None:
    """Save split DataFrames and binarizer."""
    train_path = ARTIFACTS_DIR / "train_df.csv"
    test_path = ARTIFACTS_DIR / "test_df.csv"
    binarizer_path = ARTIFACTS_DIR / "label_binarizer.pkl"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    with open(binarizer_path, "wb") as f:
        pickle.dump(binarizer, f)

    print(f"✅ Artifacts saved to {ARTIFACTS_DIR}")
    print(f"   - {train_path.name} ({len(train_df)} samples)")
    print(f"   - {test_path.name} ({len(test_df)} samples)")
    print(f"   - {binarizer_path.name}")


def main():
    print(f"## ⬇️ 1. Downloading dataset ({DATASET_REF})")
    try:
        read_only_path = kagglehub.dataset_download(DATASET_REF)
        print(f"✅ Dataset found (read-only) at: {read_only_path}")

        # Clean and copy to writable directory
        directory_root = consolidate_dataset(read_only_path)
    except Exception as e:
        print(f"❌ Critical error: {e}")
        return

    print("\n## 🗂️ 3. Indexing images and creating splits")
    dataset_root = Path(directory_root)
    df, class_counts = build_dataframe(dataset_root)
    print(f"   -> Found {len(df)} images across {len(class_counts)} classes.")

    plot_class_distribution_global(class_counts)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    # Fit the binarizer on train only to avoid leakage
    binarizer = LabelBinarizer()
    binarizer.fit(train_df["label"])

    save_artifacts(train_df, test_df, binarizer)

    # Split distribution
    print_class_distribution(train_df["label"].values, "Train split")
    print_class_distribution(test_df["label"].values, "Test split")


if __name__ == "__main__":
    main()
