"""
backend/main.py — KidneyAI FastAPI application entry point.
Run: uvicorn backend.main:app --reload --port 8000
"""
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.config import settings
from backend.database import create_tables
from backend.services.model_manager import model_manager
from backend.routes import prediction, history, model_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kidneyai")

# Create FastAPI app
app = FastAPI(
    title="KidneyAI API",
    description=(
        "Explainable AI-Based Kidney Condition Classification API. "
        "For research and educational purposes only."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    logger.info("KidneyAI Backend starting up...")
    create_tables()
    model_manager.initialize()
    status = model_manager.model_status()
    logger.info(f"Model status: {status}")
    if not any(status.values()):
        logger.warning(
            "No trained models found. "
            "Run ml/train_cnn.py, ml/train_resnet.py, ml/extract_resnet_features.py, "
            "and ml/train_random_forest.py to train models."
        )
    logger.info("KidneyAI Backend ready.")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


# Include routers
app.include_router(prediction.router)
app.include_router(history.router)
app.include_router(model_info.router)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "KidneyAI API v2.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "disclaimer": settings.DISCLAIMER,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
