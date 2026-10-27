"""
ml/train_cnn.py — Train custom CNN for kidney condition classification.
Run: python -m ml.train_cnn
"""
import os, json, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.config import (DATASET_PATH, CNN_DIR, CNN_MODEL_PATH, CNN_HISTORY_PATH,
                       CNN_METRICS_PATH, CNN_CLASS_INDICES_PATH, CNN_VERSION_PATH,
                       CNN_EPOCHS, BATCH_SIZE, RANDOM_SEED, NUM_CLASSES, CLASS_NAMES)
from ml.preprocessing import load_and_split_dataset

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def build_cnn(input_shape=(150, 150, 3), num_classes=3) -> tf.keras.Model:
    """Build improved custom CNN with BatchNormalization and GlobalAveragePooling."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='last_conv'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ], name='KidneyAI_CNN')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def make_dataset(data_list, batch_size, augment=False, shuffle=True):
    """Build tf.data.Dataset from (path, label) list."""
    paths = [d[0] for d in data_list]
    labels = [d[1] for d in data_list]

    def load_and_preprocess(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [150, 150])
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    def augment_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.image.random_saturation(img, 0.9, 1.1)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 5000), seed=RANDOM_SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    print("=" * 60)
    print("KidneyAI — Training Custom CNN")
    print("=" * 60)

    train_data, val_data, test_data = load_and_split_dataset(DATASET_PATH, random_seed=RANDOM_SEED)

    train_ds = make_dataset(train_data, BATCH_SIZE, augment=True)
    val_ds   = make_dataset(val_data,   BATCH_SIZE, augment=False, shuffle=False)
    test_ds  = make_dataset(test_data,  BATCH_SIZE, augment=False, shuffle=False)

    # Compute class weights for imbalanced dataset
    train_labels = [d[1] for d in train_data]
    cw = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = {i: float(w) for i, w in enumerate(cw)}
    print(f"Class weights: {class_weights}")

    model = build_cnn(num_classes=NUM_CLASSES)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(CNN_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
    ]

    print("\nTraining...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CNN_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Save history (convert numpy floats to Python floats)
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(CNN_HISTORY_PATH, 'w') as f:
        json.dump(hist_dict, f, indent=2)

    # Save class indices
    class_indices = {name: i for i, name in enumerate(CLASS_NAMES)}
    with open(CNN_CLASS_INDICES_PATH, 'w') as f:
        json.dump(class_indices, f, indent=2)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_true = [d[1] for d in test_data]
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)))
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "accuracy": float(acc),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
            }
            for i in range(NUM_CLASSES)
        },
        "confusion_matrix": cm,
        "test_size": len(y_true),
        "class_names": CLASS_NAMES,
        "note": "Evaluated on held-out test set (15% of dataset, image-level stratified split).",
    }
    with open(CNN_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(CNN_VERSION_PATH, 'w') as f:
        json.dump({"version": "1.0.0", "input_size": "150x150", "num_classes": NUM_CLASSES}, f, indent=2)

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Metrics saved to: {CNN_METRICS_PATH}")
    print(f"Model saved to:   {CNN_MODEL_PATH}")


if __name__ == "__main__":
    main()
