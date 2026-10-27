"""
backend/routes/history.py — Analysis history endpoints.
"""
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from backend.database import get_db, Analysis
from backend.schemas.prediction_schema import AnalysisList, HistoryItem, AnalysisDetail

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/api/history", response_model=AnalysisList, tags=["History"])
def get_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    prediction: str = Query(None, description="Filter by prediction class"),
    db: Session = Depends(get_db),
):
    """Return paginated analysis history."""
    query = db.query(Analysis)
    if prediction:
        query = query.filter(Analysis.prediction == prediction)
    total = query.count()
    records = query.order_by(Analysis.created_at.desc()).offset(skip).limit(limit).all()

    items = [
        HistoryItem(
            analysis_id=r.analysis_id,
            created_at=r.created_at,
            prediction=r.prediction,
            probability=r.probability,
            model_name=r.model_name,
            image_filename=r.image_filename,
        )
        for r in records
    ]
    return AnalysisList(total=total, items=items)


@router.get("/api/history/{analysis_id}", response_model=AnalysisDetail, tags=["History"])
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    """Return details for a single analysis."""
    record = db.query(Analysis).filter(Analysis.analysis_id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    all_probs = None
    if record.all_probabilities_json:
        try:
            all_probs = json.loads(record.all_probabilities_json)
        except Exception:
            pass

    return AnalysisDetail(
        analysis_id=record.analysis_id,
        created_at=record.created_at,
        prediction=record.prediction,
        probability=record.probability,
        model_name=record.model_name,
        model_version=record.model_version or "unknown",
        explanation_available=record.explanation_available,
        all_probabilities=all_probs,
    )


@router.delete("/api/history/{analysis_id}", tags=["History"])
def delete_analysis(analysis_id: str, db: Session = Depends(get_db)):
    """Delete an analysis record."""
    record = db.query(Analysis).filter(Analysis.analysis_id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    db.delete(record)
    db.commit()
    logger.info(f"Deleted analysis: {analysis_id}")
    return {"message": f"Analysis {analysis_id} deleted."}
