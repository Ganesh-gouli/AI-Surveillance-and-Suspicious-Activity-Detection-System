import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base, Camera, Incident, RestrictedZone, User

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sentinelvision.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Initializes clean SQLite database schema without fake or synthetic data.
    """
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Default Primary Camera Channel if database is empty
        if db.query(Camera).count() == 0:
            default_cam = Camera(
                id="CAM-01",
                name="Primary Surveillance Channel",
                location="Main Observation Area",
                stream_url="webcam://0",
                resolution="1920x1080",
                fps=30,
                status="ONLINE",
                ai_confidence=0.0,
                people_detected=0,
                active_threats=0
            )
            db.add(default_cam)
            db.commit()
    except Exception as e:
        print(f"[Database] Initialization note: {e}")
    finally:
        db.close()
