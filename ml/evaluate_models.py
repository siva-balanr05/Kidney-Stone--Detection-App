"""
ml/evaluate_models.py — Compare all three models on the shared test set.
Run: python -m ml.evaluate_models
"""
import os, json, sys
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.config import (DATASET_PATH, MODEL_ARTIFACTS_DIR, CLASS_NAMES, BATCH_SIZE, RANDOM_SEED,
                       CNN_MODEL_PATH, RESNET_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH,
                       RF_TEST_FEATURES_PATH, RF_TEST_LABELS_PATH, COMPARISON_PATH)
from ml.preprocessing import load_and_split_dataset


def make_cnn_ds(test_data, batch_size):
    paths = [d[0] for d in test_data]
    labels = [d[1] for d in test_data]
    def load(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [150, 150])
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_resnet_ds(test_data, batch_size):
    paths = [d[0] for d in test_data]
    labels = [d[1] for d in test_data]
    def load(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img, label
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def main():
    print("=" * 60)
    print("KidneyAI — Model Evaluation Comparison")
    print("=" * 60)

    _, _, test_data = load_and_split_dataset(DATASET_PATH, random_seed=RANDOM_SEED)
    y_true = [d[1] for d in test_data]
    results = {}
    eval_dir = os.path.join(MODEL_ARTIFACTS_DIR, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # --- CNN ---
    if os.path.exists(CNN_MODEL_PATH):
        print("\nEvaluating Custom CNN...")
        cnn = tf.keras.models.load_model(CNN_MODEL_PATH)
        test_ds = make_cnn_ds(test_data, BATCH_SIZE)
        y_pred_probs = cnn.predict(test_ds)
        y_pred = np.argmax(y_pred_probs, axis=1)
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
        cm = confusion_matrix(y_true, y_pred).tolist()
        results["CNN"] = {"report": report, "confusion_matrix": cm}
        print(f"CNN Accuracy: {report['accuracy']:.4f}")
    else:
        results["CNN"] = {"status": "model_not_found", "path": CNN_MODEL_PATH}

    # --- ResNet ---
    if os.path.exists(RESNET_MODEL_PATH):
        print("\nEvaluating ResNet50...")
        resnet = tf.keras.models.load_model(RESNET_MODEL_PATH)
        test_ds = make_resnet_ds(test_data, BATCH_SIZE)
        y_pred_probs = resnet.predict(test_ds)
        y_pred = np.argmax(y_pred_probs, axis=1)
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
        cm = confusion_matrix(y_true, y_pred).tolist()
        results["ResNet50"] = {"report": report, "confusion_matrix": cm}
        print(f"ResNet Accuracy: {report['accuracy']:.4f}")
    else:
        results["ResNet50"] = {"status": "model_not_found", "path": RESNET_MODEL_PATH}

    # --- RF ---
    if os.path.exists(RF_MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(RF_TEST_FEATURES_PATH):
        print("\nEvaluating Random Forest...")
        rf = joblib.load(RF_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        X_test_labels = np.load(RF_TEST_LABELS_PATH)
        X_test_feats = np.load(RF_TEST_FEATURES_PATH)
        X_test_scaled = scaler.transform(X_test_feats)
        y_pred = rf.predict(X_test_scaled)
        y_true_rf = X_test_labels.tolist()
        report = classification_report(y_true_rf, y_pred, target_names=CLASS_NAMES, output_dict=True)
        cm = confusion_matrix(y_true_rf, y_pred).tolist()
        results["Random_Forest"] = {"report": report, "confusion_matrix": cm}
        print(f"RF Accuracy: {report['accuracy']:.4f}")
    else:
        results["Random_Forest"] = {"status": "model_or_features_not_found"}

    results["meta"] = {
        "class_names": CLASS_NAMES,
        "test_size": len(y_true),
        "note": "All models evaluated on the same held-out test set (15% of dataset).",
    }

    with open(COMPARISON_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=to_serializable)

    print(f"\nComparison saved to: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
