from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from .config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Exam(Base):
    __tablename__ = "exams"
    id = Column(Integer, primary_key=True)
    patient_ref = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="pending")      # pending | done | error
    predictions = Column(JSON, default=dict)          # {label: prob}
    gradcam = Column(JSON, default=dict)              # {label: png path}
    report_path = Column(String, default="")


def init_db():
    Base.metadata.create_all(engine)
