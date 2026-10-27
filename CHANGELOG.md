# CHANGELOG.md — KidneyAI

## [2.0.0] — 2026-09-22 — Major Rewrite

### Added

#### Frontend (New)
- React + Vite application replacing Streamlit
- 8 pages: Landing, Dashboard, Analyze, Result, History, Model Performance, Model Info, About
- Professional dark medical-AI design system
- Drag-and-drop image upload with react-dropzone
- Real-time Grad-CAM heatmap visualization
- Multi-model prediction comparison UI
- Analysis history with search, filter, sort
- Model performance metrics visualization
- Responsive layout with collapsible sidebar

#### Backend (New)
- FastAPI REST API replacing Streamlit embedded logic
- POST /api/predict endpoint with full ML pipeline
- GET /api/history, GET /api/history/{id}, DELETE /api/history/{id}
- GET /api/model-info with real model status
- GET /api/health health check
- SQLite database via SQLAlchemy for analysis history
- Pydantic v2 request/response schemas
- Global error handling middleware
- CORS configuration
- Swagger UI at /docs

#### ML Pipeline (Extended)
- Improved Custom CNN: added BatchNormalization, GlobalAveragePooling, 4th conv block
- ResNet50 transfer learning with frozen base + trainable head
- ResNet50 feature extraction (2048-dim vectors)
- Random Forest classifier on ResNet features
- Proper stratified train/val/test split (70/15/15)
- EarlyStopping and ModelCheckpoint
- Class weight balancing for imbalanced dataset
- Grad-CAM implementation for CNN and ResNet
- Evaluation script comparing all three models
- Reproducible training with fixed random seeds

#### Documentation
- PROJECT_AUDIT.md — comprehensive repository audit
- ARCHITECTURE.md — system and data flow architecture
- MODEL_DOCUMENTATION.md — ML model details
- CHANGELOG.md (this file)
- README.md — complete rewrite with setup instructions

#### Infrastructure
- requirements.txt
- model_artifacts/ directory structure
- backend/uploads/ for temporary image storage
- .env.example for environment configuration
- SQLite database for history persistence

### Modified
- train_model.py: kept for reference; replaced by ml/ scripts
- README.md: completely rewritten

### Fixed
- CRITICAL: Class label order bug — Cyst/Normal/Tumor now correctly mapped to argmax indices 0/1/2
- CRITICAL: Hardcoded absolute paths removed; all paths configurable
- CRITICAL: No model file crashes — all services handle missing models gracefully
- HIGH: No error handling — all API endpoints have proper 400/503/500 responses
- HIGH: Clinical treatment recommendations removed — replaced with research disclaimer
- MEDIUM: No medical disclaimer — added to every result response and every page

### Known Limitations
- Dataset split is image-level (no patient IDs available)
- No HTTPS in development mode
- Large model files (ResNet50 ~100MB) not included in git; training required
- Grad-CAM not available for Random Forest (documented and labeled)
- Model probabilities are uncalibrated softmax outputs, not clinical confidence

### Breaking Changes
- Streamlit app2.py is preserved but superseded by the FastAPI+React stack
- Model saved at C:\ML_Model\ is superseded by model_artifacts/ directory
- Dataset path is now configured via ml/config.py, not hardcoded

## [1.0.0] — Original Release

### Initial Features (Preserved)
- Streamlit single-page application (app2.py)
- Basic CNN training script (train_model.py)
- Image upload and prediction (Cyst/Normal/Tumor)
- Confidence percentage display
- Text-based treatment recommendations
