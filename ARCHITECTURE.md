# ARCHITECTURE.md — KidneyAI

## System Architecture

```
                         USER
                           │
                           ▼
              ┌─────────────────────────┐
              │   React + Vite Frontend  │
              │   (localhost:5173)        │
              │                         │
              │  Pages:                 │
              │  ├── Landing Page       │
              │  ├── Dashboard          │
              │  ├── Analyze Image      │
              │  ├── Result View        │
              │  ├── History            │
              │  ├── Model Performance  │
              │  ├── Model Info         │
              │  └── About / Research   │
              └────────────┬────────────┘
                           │ HTTP REST (Axios)
                           │ POST /api/predict
                           │ GET /api/model-info
                           │ GET /api/history
                           ▼
              ┌─────────────────────────┐
              │   FastAPI Backend        │
              │   (localhost:8000)       │
              │                         │
              │  Routes:                │
              │  ├── /api/predict       │
              │  ├── /api/history       │
              │  ├── /api/model-info    │
              │  └── /api/health        │
              │                         │
              │  Services:              │
              │  ├── Model Manager      │
              │  ├── Preprocessing Svc  │
              │  ├── CNN Service        │
              │  ├── ResNet Service     │
              │  ├── RF Service         │
              │  ├── Grad-CAM Service   │
              │  └── Prediction Svc     │
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              │       ML Pipeline        │
              │                         │
              │  ┌──────────┐            │
              │  │ Custom   │            │
              │  │ CNN      │            │
              │  │ (150×150)│            │
              │  └────┬─────┘            │
              │       │                 │
              │  ┌────┴──────────────┐  │
              │  │  ResNet50          │  │
              │  │  Transfer Learning │  │
              │  │  (224×224)         │  │
              │  └────┬──────────────┘  │
              │       │ Feature Vector  │
              │       │ (2048-dim)      │
              │  ┌────┴──────────────┐  │
              │  │  Random Forest     │  │
              │  │  (scikit-learn)    │  │
              │  └────┬──────────────┘  │
              │       │                 │
              │  ┌────┴──────────────┐  │
              │  │  Grad-CAM          │  │
              │  │  (CNN / ResNet)    │  │
              │  └───────────────────┘  │
              └─────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │   SQLite Database        │
              │   (kidneyai.db)          │
              └─────────────────────────┘
              ┌────────────┬────────────┐
              │   Model Artifacts        │
              │  cnn/kidney_cnn.keras   │
              │  resnet/kidney_resnet.keras │
              │  random_forest/rf_model.joblib │
              └─────────────────────────┘
```

## Frontend Architecture

```
frontend/
├── index.html              # HTML entry, Google Fonts
├── vite.config.js          # Vite config + API proxy
├── src/
│   ├── main.jsx            # React root mount
│   ├── App.jsx             # Router + Layout
│   ├── index.css           # Global design system
│   ├── services/
│   │   └── api.js          # Axios API service
│   ├── hooks/
│   │   └── useApi.js       # Data fetching hook
│   ├── components/
│   │   ├── Sidebar.jsx     # Navigation sidebar
│   │   ├── Layout.jsx      # Page layout wrapper
│   │   ├── Toast.jsx       # Notification toast
│   │   ├── MetricCard.jsx  # Stat display card
│   │   ├── ModelStatusBadge.jsx
│   │   ├── ConfusionMatrix.jsx
│   │   └── SkeletonLoader.jsx
│   └── pages/
│       ├── LandingPage.jsx
│       ├── Dashboard.jsx
│       ├── AnalyzePage.jsx
│       ├── ResultPage.jsx
│       ├── HistoryPage.jsx
│       ├── ModelPerformancePage.jsx
│       ├── ModelInfoPage.jsx
│       └── AboutPage.jsx
```

## Backend Architecture

```
backend/
├── main.py                 # FastAPI app entry
├── config.py               # Settings (pydantic-settings)
├── database.py             # SQLAlchemy + SQLite
├── routes/
│   ├── prediction.py       # POST /api/predict
│   ├── history.py          # CRUD /api/history
│   └── model_info.py       # GET /api/model-info, /health
├── services/
│   ├── model_manager.py    # Singleton model loader
│   ├── preprocessing_service.py
│   ├── cnn_service.py
│   ├── resnet_service.py
│   ├── random_forest_service.py
│   ├── gradcam_service.py
│   └── prediction_service.py
└── schemas/
    └── prediction_schema.py
```

## ML Pipeline Architecture

```
ml/
├── config.py              # All hyperparameters
├── preprocessing.py       # Data loading + splitting
├── train_cnn.py           # Experiment A: Custom CNN
├── train_resnet.py        # Experiment B: ResNet50
├── extract_resnet_features.py  # For RF pipeline
├── train_random_forest.py # Experiment C: ResNet + RF
└── evaluate_models.py     # Model comparison

model_artifacts/
├── cnn/
│   ├── kidney_cnn.keras
│   ├── cnn_history.json
│   ├── cnn_metrics.json
│   ├── class_indices.json
│   └── version.json
├── resnet/
│   ├── kidney_resnet.keras
│   ├── resnet_history.json
│   ├── resnet_metrics.json
│   └── version.json
├── random_forest/
│   ├── rf_model.joblib
│   ├── scaler.joblib
│   ├── rf_metrics.json
│   ├── train_features.npy
│   └── version.json
└── evaluation/
    └── comparison.json
```

## Data Flow: Image Prediction

```
1. User drags/drops image in React UI
2. React validates: file type (jpg/png), size (<10MB)
3. Axios POST /api/predict (multipart/form-data)
4. FastAPI: file validation (extension, magic bytes, dimensions)
5. preprocessing_service.py:
   - Load image with Pillow
   - Preprocess for CNN (150×150, /255)
   - Preprocess for ResNet (224×224, resnet50.preprocess_input)
6. model_manager.py provides loaded models
7. cnn_service.py: CNN forward pass → [p_cyst, p_normal, p_tumor]
8. resnet_service.py: ResNet50 forward pass → [p_cyst, p_normal, p_tumor]
9. random_forest_service.py:
   - ResNet feature extraction (2048-dim)
   - StandardScaler transform
   - RF predict_proba → [p_cyst, p_normal, p_tumor]
10. gradcam_service.py: Grad-CAM on CNN (last conv layer)
    - Generate heatmap
    - Overlay on original image
    - Encode as base64 JPEG
11. prediction_service.py assembles response
12. Save to SQLite via database.py
13. Return PredictionResponse JSON
14. React navigates to /result/:analysisId
15. ResultPage displays: prediction, probabilities, Grad-CAM
```

## Class Label Order (CRITICAL)

All models use alphabetically sorted class labels (Keras flow_from_directory default):
- Index 0: Cyst
- Index 1: Normal
- Index 2: Tumor

This ordering is stored in class_indices.json and used consistently across all services.

## Grad-CAM Data Flow

```
CNN Model
    │
    ├── Forward pass for target class
    ├── Last conv layer: 'conv2d_3'
    ├── Compute gradients w.r.t. last conv activations
    ├── Global average pool gradients → channel weights
    ├── Weighted sum of activations → heatmap
    ├── ReLU + normalize to [0, 1]
    ├── Resize to original image dimensions
    ├── Apply colormap (jet)
    ├── Overlay on original image (alpha=0.4)
    └── Encode as base64 JPEG string
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/predict | Upload image, get prediction |
| GET | /api/history | List analyses (paginated) |
| GET | /api/history/{id} | Single analysis |
| DELETE | /api/history/{id} | Delete analysis |
| GET | /api/model-info | All model statuses |
| GET | /api/health | Health check |
| GET | /docs | FastAPI Swagger UI |

## Dataset Details

| Class | Images | % |
|-------|--------|---|
| Cyst | 3,709 | 33.5% |
| Normal | 5,077 | 45.9% |
| Tumor | 2,283 | 20.6% |
| **Total** | **11,069** | **100%** |

Split strategy (stratified, no patient IDs available):
- Train: 70% (7,748 images)
- Validation: 15% (1,660 images)
- Test: 15% (1,661 images)

**Limitation:** Images are split randomly, not by patient. Patient-level split is not possible as patient identifiers are absent from the dataset. This is documented as a limitation.
