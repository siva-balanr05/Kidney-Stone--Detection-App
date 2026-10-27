"""
ml/config.py — Central configuration for KidneyAI ML pipeline.
All hyperparameters, paths, and constants defined here.
"""
import os

# ─── Base Paths ────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(ROOT_DIR, "dataset", "Train")
MODEL_ARTIFACTS_DIR = os.path.join(ROOT_DIR, "model_artifacts")

# ─── Model artifact sub-directories ────────────────────────────────────────────
CNN_DIR = os.path.join(MODEL_ARTIFACTS_DIR, "cnn")
RESNET_DIR = os.path.join(MODEL_ARTIFACTS_DIR, "resnet")
RF_DIR = os.path.join(MODEL_ARTIFACTS_DIR, "random_forest")
EVAL_DIR = os.path.join(MODEL_ARTIFACTS_DIR, "evaluation")

# ─── Model file paths ───────────────────────────────────────────────────────────
CNN_MODEL_PATH = os.path.join(CNN_DIR, "kidney_cnn.keras")
CNN_HISTORY_PATH = os.path.join(CNN_DIR, "cnn_history.json")
CNN_METRICS_PATH = os.path.join(CNN_DIR, "cnn_metrics.json")
CNN_CLASS_INDICES_PATH = os.path.join(CNN_DIR, "class_indices.json")
CNN_VERSION_PATH = os.path.join(CNN_DIR, "version.json")

RESNET_MODEL_PATH = os.path.join(RESNET_DIR, "kidney_resnet.keras")
RESNET_HISTORY_PATH = os.path.join(RESNET_DIR, "resnet_history.json")
RESNET_METRICS_PATH = os.path.join(RESNET_DIR, "resnet_metrics.json")
RESNET_VERSION_PATH = os.path.join(RESNET_DIR, "version.json")

RF_MODEL_PATH = os.path.join(RF_DIR, "rf_model.joblib")
SCALER_PATH = os.path.join(RF_DIR, "scaler.joblib")
RF_METRICS_PATH = os.path.join(RF_DIR, "rf_metrics.json")
RF_TRAIN_FEATURES_PATH = os.path.join(RF_DIR, "train_features.npy")
RF_VAL_FEATURES_PATH = os.path.join(RF_DIR, "val_features.npy")
RF_TEST_FEATURES_PATH = os.path.join(RF_DIR, "test_features.npy")
RF_TRAIN_LABELS_PATH = os.path.join(RF_DIR, "train_labels.npy")
RF_VAL_LABELS_PATH = os.path.join(RF_DIR, "val_labels.npy")
RF_TEST_LABELS_PATH = os.path.join(RF_DIR, "test_labels.npy")
RF_VERSION_PATH = os.path.join(RF_DIR, "version.json")

COMPARISON_PATH = os.path.join(EVAL_DIR, "comparison.json")

# ─── Dataset classes (alphabetical = flow_from_directory order) ─────────────────
CLASS_NAMES = ["Cyst", "Normal", "Tumor"]
NUM_CLASSES = len(CLASS_NAMES)
# Maps class name → index (argmax index → class name)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

# ─── Image sizes ────────────────────────────────────────────────────────────────
IMG_SIZE_CNN = (150, 150)
IMG_SIZE_RESNET = (224, 224)

# ─── Training hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE = 32
RANDOM_SEED = 42
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

CNN_EPOCHS = 30
CNN_LR = 1e-3
CNN_PATIENCE = 5

RESNET_HEAD_EPOCHS = 20
RESNET_FINETUNE_EPOCHS = 10
RESNET_HEAD_LR = 1e-3
RESNET_FINETUNE_LR = 1e-5
RESNET_UNFREEZE_LAYERS = 30

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_SPLIT = 2

# ─── Ensure directories exist ───────────────────────────────────────────────────
for _dir in [CNN_DIR, RESNET_DIR, RF_DIR, EVAL_DIR]:
    os.makedirs(_dir, exist_ok=True)
