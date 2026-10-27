# KidneyAI — Kidney CT Scan Abnormality Detection & Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.2-61DAFB.svg?logo=react)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-8.3-646CFF.svg?logo=vite)](https://vitejs.dev)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg?logo=tensorflow)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

KidneyAI is a full-stack, AI-driven medical imaging analysis system designed to classify kidney computed tomography (CT) scans into three distinct clinical classes: **Cyst**, **Normal**, and **Tumor**.

The platform combines multi-model deep learning (Custom CNN, ResNet50 Transfer Learning, and Random Forest), visual explainability via **Grad-CAM**, a high-performance **FastAPI** backend, and a modern **React + Vite** clinical dashboard.

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Model Portfolio](#model-portfolio)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [1. Backend Setup](#1-backend-setup)
  - [2. Frontend Setup](#2-frontend-setup)
- [Training & Evaluation](#training--evaluation)
- [API Documentation](#api-documentation)
- [Running Tests](#running-tests)
- [Medical Disclaimer](#medical-disclaimer)
- [License](#license)

---

## Features

- **Multi-Model Diagnosis Support**:
  - **Custom CNN**: Lightweight sequential CNN (150×150) optimized for fast CPU inference.
  - **ResNet50 Transfer Learning**: Deep residual network fine-tuned on kidney CT scans (224×224).
  - **ResNet50 + Random Forest**: Hybrid ensemble combining deep feature extraction with tree-based classification.
- **Explainable AI (Grad-CAM)**: Generates class-activation heatmaps directly over the CT scan to highlight the anatomical regions influencing predictions.
- **Full-Stack Clinical Dashboard**:
  - Drag-and-drop CT scan upload supporting DICOM/JPEG/PNG.
  - Interactive prediction breakdowns with probability distributions and risk indicators.
  - Model benchmarking page showing accuracy, precision, recall, F1, and confusion matrices.
  - Scan history log with search, filters, and detailed diagnostic modal views.
- **Production Architecture**:
  - Asynchronous REST API powered by FastAPI with Pydantic v2 validation.
  - Persistent scan logging via SQLite and SQLAlchemy.
  - Responsive, dark-mode medical UI with React 19, Lucide icons, and Recharts.

---

## System Architecture

```
                       User (Browser)
                             │
                             ▼
               ┌──────────────────────────┐
               │   React + Vite Frontend  │
               │      (Port: 5173)        │
               │                          │
               │ • Landing & Dashboard    │
               │ • Scan Analysis & Upload │
               │ • Grad-CAM Visualization │
               │ • Model Comparison View  │
               │ • Historical Records     │
               └─────────────┬────────────┘
                             │ HTTP / JSON (Axios)
                             ▼
               ┌──────────────────────────┐
               │     FastAPI Backend      │
               │      (Port: 8000)        │
               │                          │
               │ • /api/predict           │
               │ • /api/history           │
               │ • /api/model-info        │
               │ • /api/health            │
               └─────────────┬────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌──────────────────────────┐     ┌──────────────────────────┐
│       ML Pipelines       │     │     Storage & Data       │
│                          │     │                          │
│ • Custom CNN             │     │ • SQLite DB (History)    │
│ • ResNet50 Deep Model    │     │ • Uploads Storage        │
│ • ResNet + Random Forest │     │ • Model Artifacts (JSON, │
│ • Grad-CAM Generator     │     │   .keras, .joblib)       │
└──────────────────────────┘     └──────────────────────────┘
```

---

## Model Portfolio

All models are trained and validated on a curated CT dataset of **11,069 images** partitioned into an image-level stratified split (70% Train, 15% Validation, 15% Test).

| Model | Input Shape | Primary Advantage | Grad-CAM Support |
|---|---|---|---|
| **Custom CNN** | 150 × 150 × 3 | Fast inference, lightweight (~5.5 MB) | Yes |
| **ResNet50** | 224 × 224 × 3 | High representational capacity, deep features | Yes |
| **Random Forest** | 2048-dim vector | Robust non-linear decision boundaries | No (Tree-based) |

---

## Project Structure

```
Kidney-Stone--Detection-App/
├── backend/                  # FastAPI Application
│   ├── config.py             # App configurations & file paths
│   ├── database.py           # SQLite & SQLAlchemy models
│   ├── main.py               # Application entry point & CORS
│   ├── routes/               # API endpoint routers (predict, history, model-info)
│   ├── schemas/              # Pydantic request/response schemas
│   ├── services/             # Preprocessing, ModelManager, Grad-CAM, Prediction
│   └── uploads/              # Local storage for scan uploads
├── frontend/                 # React 19 + Vite Frontend
│   ├── src/
│   │   ├── components/       # Layout, Navigation, Toasts
│   │   ├── pages/            # Landing, Analyze, Dashboard, History, Performance
│   │   ├── services/         # Axios API client
│   │   └── App.jsx           # Routing & application shell
│   ├── package.json          # Frontend dependencies
│   └── vite.config.js        # Vite bundler configuration
├── ml/                       # Machine Learning Pipelines
│   ├── config.py             # Hyperparameters & dataset paths
│   ├── preprocessing.py      # Dataset loading, validation, and splitting
│   ├── train_cnn.py          # Custom CNN training script
│   ├── train_resnet.py       # ResNet50 transfer learning script
│   ├── extract_resnet_features.py # Feature extraction for Random Forest
│   ├── train_random_forest.py# Random Forest classifier training
│   └── evaluate_models.py    # Multi-model evaluation & benchmark generator
├── model_artifacts/          # Serialized models, weights, & metrics
│   ├── cnn/                  # CNN model weights, history, and metrics
│   ├── resnet/               # ResNet history, metrics, and version info
│   ├── random_forest/        # RF joblib model, scaler, and metrics
│   └── evaluation/           # Benchmark comparison metrics
├── tests/                    # Automated Test Suite
│   └── test_backend.py       # API and unit test coverage
├── ARCHITECTURE.md           # Detailed technical architecture specification
├── MODEL_DOCUMENTATION.md    # In-depth ML model documentation
├── CHANGELOG.md              # Version release history
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Getting Started

### Prerequisites

- **Python**: Version 3.11 or higher
- **Node.js**: Version 18.x or higher
- **Git**

### 1. Backend Setup

1. Open a terminal in the project root directory:
   ```bash
   cd D:/Kidney-Stone--Detection-App
   ```

2. Create and activate a Python virtual environment:
   ```bash
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the FastAPI server:
   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

   The backend will be available at:
   - API Server: `http://localhost:8000`
   - Interactive Swagger Docs: `http://localhost:8000/docs`
   - Redoc: `http://localhost:8000/redoc`

### 2. Frontend Setup

1. Open a new terminal in the `frontend` directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the Vite development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to `http://localhost:5173`.

---

## Training & Evaluation

To train models from scratch or regenerate evaluation metrics using your local dataset:

```bash
# 1. Train Custom CNN
python -m ml.train_cnn

# 2. Train ResNet50
python -m ml.train_resnet

# 3. Extract Features and Train Random Forest
python -m ml.extract_resnet_features
python -m ml.train_random_forest

# 4. Generate Comparative Benchmark Report
python -m ml.evaluate_models
```

Trained models and metrics are saved directly into `model_artifacts/`.

---

## API Documentation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check & service readiness |
| `POST` | `/api/predict` | Upload a CT scan image and receive classification + Grad-CAM |
| `GET` | `/api/history` | Retrieve paginated history of past scans and predictions |
| `GET` | `/api/history/{id}` | Retrieve details and heatmaps for a specific prediction |
| `GET` | `/api/model-info` | Metadata and performance metrics for all available models |

---

## Running Tests

Run the test suite using `pytest`:

```bash
pytest tests/ -v
```

---

## Medical Disclaimer

> [!WARNING]
> **RESEARCH & EDUCATIONAL USE ONLY**
> 
> KidneyAI is designed solely as an investigational, educational, and research software tool. It is **NOT** a medical device, is **NOT** certified for diagnostic use by any regulatory authority (e.g., US FDA, CE mark), and must **NEVER** replace professional medical consultation, clinical diagnosis, or treatment decisions. Always consult a qualified radiologist or physician for medical concerns.

---

## License

This project is licensed under the [MIT License](LICENSE).
