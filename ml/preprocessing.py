"""
ml/preprocessing.py — Data loading and preprocessing utilities for KidneyAI.
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.config import CLASS_NAMES, DATASET_PATH, RANDOM_SEED, TEST_SPLIT, VAL_SPLIT


def preprocess_for_cnn(img_array: np.ndarray) -> np.ndarray:
    """Preprocess image array for the custom CNN: resize to 150x150, normalize to [0,1]."""
    img_resized = cv2.resize(img_array, (150, 150))
    return img_resized.astype(np.float32) / 255.0


def preprocess_for_resnet(img_array: np.ndarray) -> np.ndarray:
    """Preprocess image for ResNet50: resize to 224x224, apply ResNet preprocessing."""
    img_resized = cv2.resize(img_array, (224, 224))
    return tf.keras.applications.resnet50.preprocess_input(img_resized.astype(np.float32))


def load_and_split_dataset(
    dataset_path: str = DATASET_PATH,
    test_split: float = TEST_SPLIT,
    val_split: float = VAL_SPLIT,
    random_seed: int = RANDOM_SEED,
) -> tuple:
    """
    Load all image paths and labels from dataset_path subdirectories.
    Split into train, val, test using stratified sampling.
    
    Returns:
        train_data: list of (path, label_int) tuples
        val_data:   list of (path, label_int) tuples
        test_data:  list of (path, label_int) tuples
    
    Class indices (alphabetical, matching Keras flow_from_directory):
        Cyst   -> 0
        Normal -> 1
        Tumor  -> 2
    """
    paths = []
    labels = []

    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"WARNING: class directory not found: {class_dir}")
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(class_dir, filename))
                labels.append(idx)

    if not paths:
        raise FileNotFoundError(f"No images found in {dataset_path}")

    # Split: train+val vs test (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        paths, labels, test_size=test_split, random_state=random_seed, stratify=labels
    )
    # Split: train vs val (stratified within remaining data)
    relative_val_split = val_split / (1.0 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_split, random_state=random_seed, stratify=y_temp
    )

    train_data = list(zip(X_train, y_train))
    val_data = list(zip(X_val, y_val))
    test_data = list(zip(X_test, y_test))

    print(f"Dataset split — Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data
