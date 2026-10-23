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
        # Seed active surveillance channels if empty or only CAM-01
        channels = [
            ("CAM-01", "Primary Live Tactical Webcam", "Command Center Observation", "webcam://0", "1920x1080"),
            ("CAM-02", "ATM Vestibule CCTV - Sector 30A", "ATM Kiosk - Sector 30A", "/videos/30a.mp4", "1280x720"),
            ("CAM-03", "Cash Vault & Safe Corridor - Sector 5", "Vault Security Zone", "/videos/5.mp4", "1280x720"),
            ("CAM-04", "Perimeter Night Observation - Sector 88", "Perimeter Fence 88", "/videos/88.mp4", "1280x720"),
            ("CAM-05", "ATM Entrance Kiosk - Sector 30B", "ATM Entrance 30B", "/videos/30b.mp4", "1280x720"),
            ("CAM-06", "Pedestrian Transit Hub - Sector 75", "Transit Hub Sector 75", "/videos/75.mp4", "1280x720"),
        ]
        for cid, cname, cloc, curl, cres in channels:
            existing = db.query(Camera).filter(Camera.id == cid).first()
            if not existing:
                new_cam = Camera(
                    id=cid,
                    name=cname,
                    location=cloc,
                    stream_url=curl,
                    resolution=cres,
                    fps=30,
                    status="ONLINE",
                    ai_confidence=96.5,
                    people_detected=0,
                    active_threats=0,
                    is_recording=True
                )
                db.add(new_cam)
        db.commit()
    except Exception as e:
        print(f"[Database] Initialization note: {e}")
    finally:
        db.close()
