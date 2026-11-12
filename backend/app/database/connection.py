from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

# Configure engine with search_path pointing to agenticswarm schema
connect_args = {"options": "-csearch_path=agenticswarm,public"}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=False
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative Base for models
Base = declarative_base()

def get_db():
    """
    FastAPI dependency yielding database session per request.
    Ensures session is properly closed after response.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Ensures the schema exists and creates all defined database tables.
    """
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS agenticswarm;"))
        conn.commit()

    # Import models so Base registers them before create_all
    import database.models  # noqa: F401
    Base.metadata.create_all(bind=engine)
