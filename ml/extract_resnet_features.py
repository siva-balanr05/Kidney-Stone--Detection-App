"""
ml/extract_resnet_features.py — Extract ResNet50 feature vectors for Random Forest training.
Run: python -m ml.extract_resnet_features
"""
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.config import (DATASET_PATH, RF_DIR, RF_TRAIN_FEATURES_PATH, RF_VAL_FEATURES_PATH,
                       RF_TEST_FEATURES_PATH, RF_TRAIN_LABELS_PATH, RF_VAL_LABELS_PATH,
                       RF_TEST_LABELS_PATH, BATCH_SIZE, RANDOM_SEED)
from ml.preprocessing import load_and_split_dataset


def build_feature_extractor():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    return models.Model(inputs=base.input, outputs=x, name='ResNet50_FeatureExtractor')


def make_path_dataset(paths):
    def load(path):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img
    ds = tf.data.Dataset.from_tensor_slices(paths)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def extract_features(extractor, data_list):
    paths = [d[0] for d in data_list]
    labels = np.array([d[1] for d in data_list])
    ds = make_path_dataset(paths)
    features = extractor.predict(ds, verbose=1)
    return features, labels


def main():
    print("=" * 60)
    print("KidneyAI — Extracting ResNet50 Features for Random Forest")
    print("=" * 60)
    os.makedirs(RF_DIR, exist_ok=True)

    train_data, val_data, test_data = load_and_split_dataset(DATASET_PATH, random_seed=RANDOM_SEED)
    extractor = build_feature_extractor()

    print(f"\nExtracting training features ({len(train_data)} images)...")
    train_feats, train_labels = extract_features(extractor, train_data)
    np.save(RF_TRAIN_FEATURES_PATH, train_feats)
    np.save(RF_TRAIN_LABELS_PATH, train_labels)

    print(f"\nExtracting validation features ({len(val_data)} images)...")
    val_feats, val_labels = extract_features(extractor, val_data)
    np.save(RF_VAL_FEATURES_PATH, val_feats)
    np.save(RF_VAL_LABELS_PATH, val_labels)

    print(f"\nExtracting test features ({len(test_data)} images)...")
    test_feats, test_labels = extract_features(extractor, test_data)
    np.save(RF_TEST_FEATURES_PATH, test_feats)
    np.save(RF_TEST_LABELS_PATH, test_labels)

    print(f"\nDone. Feature shape: {train_feats.shape[1]}-dimensional vectors.")
    print(f"Saved to: {RF_DIR}")


if __name__ == "__main__":
    main()
