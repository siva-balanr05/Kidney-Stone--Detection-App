"""
ml/train_random_forest.py — Train Random Forest on ResNet50 features.
Run: python -m ml.train_random_forest  (after running extract_resnet_features.py)
"""
import os, json, sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.config import (RF_DIR, RF_MODEL_PATH, SCALER_PATH, RF_METRICS_PATH, RF_VERSION_PATH,
                       RF_TRAIN_FEATURES_PATH, RF_VAL_FEATURES_PATH, RF_TEST_FEATURES_PATH,
                       RF_TRAIN_LABELS_PATH, RF_TEST_LABELS_PATH,
                       RF_N_ESTIMATORS, RANDOM_SEED, NUM_CLASSES, CLASS_NAMES)


def main():
    print("=" * 60)
    print("KidneyAI — Training Random Forest on ResNet50 Features")
    print("=" * 60)

    # Check features exist
    for path in [RF_TRAIN_FEATURES_PATH, RF_TEST_FEATURES_PATH, RF_TRAIN_LABELS_PATH, RF_TEST_LABELS_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing feature file: {path}\nRun extract_resnet_features.py first.")

    X_train = np.load(RF_TRAIN_FEATURES_PATH)
    y_train = np.load(RF_TRAIN_LABELS_PATH)
    X_test  = np.load(RF_TEST_FEATURES_PATH)
    y_test  = np.load(RF_TEST_LABELS_PATH)

    print(f"Train features: {X_train.shape}, Test features: {X_test.shape}")

    # Scale features (fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Train Random Forest
    print(f"\nTraining RandomForest (n_estimators={RF_N_ESTIMATORS})...")
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_features='sqrt',
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
    )
    rf.fit(X_train_scaled, y_train)

    # Evaluate on test set
    y_pred = rf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=list(range(NUM_CLASSES)))
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": float(acc),
        "per_class": {
            CLASS_NAMES[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}
            for i in range(NUM_CLASSES)
        },
        "confusion_matrix": cm,
        "test_size": int(len(y_test)),
        "class_names": CLASS_NAMES,
        "feature_dim": int(X_train.shape[1]),
        "note": "Evaluated on held-out test set using features from ResNet50 base (ImageNet weights).",
    }

    os.makedirs(RF_DIR, exist_ok=True)
    with open(RF_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(rf, RF_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(RF_VERSION_PATH, 'w') as f:
        json.dump({"version": "1.0.0", "feature_dim": int(X_train.shape[1]), "num_classes": NUM_CLASSES}, f, indent=2)

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"RF model saved to:  {RF_MODEL_PATH}")
    print(f"Scaler saved to:    {SCALER_PATH}")


if __name__ == "__main__":
    main()
