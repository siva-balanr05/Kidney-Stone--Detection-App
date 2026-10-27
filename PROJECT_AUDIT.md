# PROJECT_AUDIT.md — KidneyAI

**Audit Date:** 2026-09-22
**Repository:** Kidney-Stone--Detection-App

---

## A. Existing Project Structure

```
Kidney-Stone--Detection-App/
├── .git/
├── README.md                      # 3-line minimal README
├── app2.py                        # Streamlit application (61 lines)
├── train_model.py                 # Basic CNN training script (51 lines)
└── dataset/
    └── Train/
        ├── Cyst/                  # 3,709 JPEG images
        ├── Normal/                # 5,077 JPEG images
        └── Tumor/                 # 2,283 JPEG images
Total dataset: 11,069 images
```

No frontend folder | No backend folder | No model artifacts | No tests | No config files

---

## B. Existing Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| Frontend | Streamlit | Python-only UI, no React/JS |
| Backend | None | Embedded in Streamlit |
| ML | TensorFlow 2.15 / Keras 2.15 | CNN model |
| Database | None | No persistence |
| API | None | No REST API |
| Auth | None | |

---

## C. Existing ML Implementation

- **Architecture:** Sequential CNN: Conv2D(32/64/128) → Flatten → Dense(128) → Dropout(0.5) → Softmax
- **Input Size:** 150×150×3
- **Classes:** Cyst, Normal, Tumor (3 classes, NOT binary stone/no-stone)
- **Model Path:** C:\ML_Model\kidney_model.keras (hardcoded — broken on all other machines)
- **Dataset Path:** C:/Users/sivab/Downloads/dataset (hardcoded — broken)
- **Bug:** class_labels in app2.py = ["Normal","Tumor","Cyst"] but flow_from_directory sorts alphabetically → ["Cyst","Normal","Tumor"]. SILENT CLASS MISMATCH BUG.
- **No test set** — only train/val split on training folder

---

## D. Existing Web Application

- Streamlit single-page app
- Image upload, display, predict, show class + confidence%
- Text-based clinical recommendations (INAPPROPRIATE — uncalibrated model)
- No history, dashboard, navigation, Grad-CAM, error handling, or medical disclaimer

---

## E. Problems and Risks

1. CRITICAL: Hardcoded absolute paths — app crashes on any other machine
2. CRITICAL: Model file not in repository — app crashes immediately
3. CRITICAL: Class label order bug — predictions mapped to wrong classes
4. HIGH: No requirements.txt — unreproducible environment
5. HIGH: No held-out test set — no reliable evaluation
6. HIGH: Clinical treatment suggestions from uncalibrated ML model
7. MEDIUM: No error handling for missing model
8. MEDIUM: No medical disclaimer
9. LOW: Unused imports, no logging, no versioning

---

## Current Architecture

```
User → Streamlit app2.py
  → Image Upload Widget
  → preprocess_image() [resize 150×150, /255]
  → model.predict() [C:\ML_Model\kidney_model.keras]
  → class_labels[argmax] [WRONG ORDER BUG]
  → Display: class, confidence%, text recommendation
```

---

## Recommended Modification Plan

Decision: EXTEND (not replace) the project.
- Keep 3-class dataset (Cyst/Normal/Tumor)
- Migrate to FastAPI backend + React frontend
- Add ResNet50 transfer learning + Random Forest
- Add Grad-CAM explainability
- Fix all bugs

### Files Modified
- train_model.py → replaced by ml/ scripts
- README.md → complete rewrite

### Files Created
- requirements.txt, backend/, ml/, frontend/, model_artifacts/
- PROJECT_AUDIT.md, ARCHITECTURE.md, MODEL_DOCUMENTATION.md, CHANGELOG.md
