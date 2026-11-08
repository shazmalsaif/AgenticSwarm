import os
from pathlib import Path
from dotenv import load_dotenv

# Locate and load .env file
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    PROJECT_NAME: str = "AgentSwarm API"
    VERSION: str = "1.0.0"
    
    # Database (loaded strictly from .env)
    DATABASE_URL: str = (os.getenv("DATABASE_URL") or "").strip()
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set. Please define it in backend/.env")
    
    # JWT & Auth
    SECRET_KEY: str = os.getenv("SECRET_KEY", "agentswarm_super_secret_jwt_key_2026_x99!a@")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "null",  # file:// origins
        "*"
    ]

settings = Settings()

