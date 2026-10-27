"""
ml/train_resnet.py — Train ResNet50 transfer learning model.
Run: python -m ml.train_resnet
"""
import os, json, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.config import (DATASET_PATH, RESNET_DIR, RESNET_MODEL_PATH, RESNET_HISTORY_PATH,
                       RESNET_METRICS_PATH, RESNET_VERSION_PATH, RESNET_HEAD_EPOCHS,
                       RESNET_FINETUNE_EPOCHS, RESNET_HEAD_LR, RESNET_FINETUNE_LR,
                       RESNET_UNFREEZE_LAYERS, BATCH_SIZE, RANDOM_SEED, NUM_CLASSES, CLASS_NAMES)
from ml.preprocessing import load_and_split_dataset

tf.random.set_seed(RANDOM_SEED)


def make_dataset(data_list, batch_size, shuffle=True):
    paths = [d[0] for d in data_list]
    labels = [d[1] for d in data_list]

    def load_and_preprocess(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 5000), seed=RANDOM_SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_resnet(num_classes=3):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base.input, outputs=out, name='KidneyAI_ResNet50')
    return model, base


def main():
    print("=" * 60)
    print("KidneyAI — Training ResNet50 Transfer Learning")
    print("=" * 60)

    train_data, val_data, test_data = load_and_split_dataset(DATASET_PATH, random_seed=RANDOM_SEED)
    train_ds = make_dataset(train_data, BATCH_SIZE)
    val_ds   = make_dataset(val_data,   BATCH_SIZE, shuffle=False)
    test_ds  = make_dataset(test_data,  BATCH_SIZE, shuffle=False)

    model, base_model = build_resnet(num_classes=NUM_CLASSES)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(RESNET_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
    ]

    print("\nPhase 1: Training classification head (base frozen)...")
    model.compile(optimizer=tf.keras.optimizers.Adam(RESNET_HEAD_LR),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history1 = model.fit(train_ds, validation_data=val_ds,
                         epochs=RESNET_HEAD_EPOCHS, callbacks=callbacks, verbose=1)

    print("\nPhase 2: Fine-tuning last layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-RESNET_UNFREEZE_LAYERS]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(RESNET_FINETUNE_LR),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history2 = model.fit(train_ds, validation_data=val_ds,
                         epochs=RESNET_FINETUNE_EPOCHS, callbacks=callbacks, verbose=1)

    # Combine histories
    full_hist = {}
    for k in history1.history:
        v1 = [float(x) for x in history1.history[k]]
        v2 = [float(x) for x in history2.history.get(k, [])]
        full_hist[k] = v1 + v2
    with open(RESNET_HISTORY_PATH, 'w') as f:
        json.dump(full_hist, f, indent=2)

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
        "per_class": {CLASS_NAMES[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])} for i in range(NUM_CLASSES)},
        "confusion_matrix": cm,
        "test_size": len(y_true),
        "class_names": CLASS_NAMES,
        "note": "Evaluated on held-out test set (15% of dataset).",
    }
    with open(RESNET_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(RESNET_VERSION_PATH, 'w') as f:
        json.dump({"version": "1.0.0", "input_size": "224x224", "num_classes": NUM_CLASSES}, f, indent=2)

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Model saved to: {RESNET_MODEL_PATH}")


if __name__ == "__main__":
    main()
