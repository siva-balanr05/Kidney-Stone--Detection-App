"""
backend/database.py — SQLAlchemy database setup for analysis history.
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.config import settings

# Ensure directory exists
db_dir = os.path.dirname(settings.DB_URL.replace("sqlite:///", ""))
os.makedirs(db_dir, exist_ok=True)

engine = create_engine(settings.DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String(36), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    prediction = Column(String(50), nullable=False)
    probability = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), default="unknown")
    image_filename = Column(String(255))
    explanation_available = Column(Boolean, default=False)
    all_probabilities_json = Column(Text)  # JSON string of {class: probability}


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
