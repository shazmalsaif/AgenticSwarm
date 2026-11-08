from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.config import settings
from app.auth import hash_password, verify_password, create_access_token, decode_access_token
from app.schemas import UserLogin, UserSignup, UserResponse, TokenResponse, MessageResponse
from database.connection import get_db, init_db
from database.models import User

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown events.
    Initializes PostgreSQL tables on startup and seeds demo account if empty.
    """
    print("[AgentSwarm] Initializing database tables...")
    try:
        init_db()
        print("[AgentSwarm] Database tables verified.")
        
        # Seed a default demo account if no users exist
        from database.connection import SessionLocal
        db = SessionLocal()
        try:
            demo_user = db.query(User).filter(User.email == "demo@agentswarm.dev").first()
            if not demo_user:
                demo_user = User(
                    email="demo@agentswarm.dev",
                    hashed_password=hash_password("password123"),
                    full_name="Lead QA Engineer"
                )
                db.add(demo_user)
                db.commit()
                print("[AgentSwarm] Demo user seeded: demo@agentswarm.dev / password123")
        finally:
            db.close()
    except Exception as e:
        print(f"[AgentSwarm] Database init warning: {e}")
        
    yield
    print("[AgentSwarm] Backend shutting down.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Configure CORS so front-end (whether opened as a file or local dev server) can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Format Pydantic 422 validation errors into clean human-readable strings.
    """
    errors = []
    for err in exc.errors():
        loc = err.get("loc", [])
        field = loc[-1] if loc else ""
        msg = err.get("msg", "Invalid input")
        if "Value error, " in msg:
            msg = msg.replace("Value error, ", "")
        errors.append(f"{field}: {msg}" if field and field != "body" else msg)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": " • ".join(errors)}
    )

# ---------------------------------------------------------------------------
# Authentication Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/auth/login", response_model=TokenResponse, summary="User Login")
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate a user by email and password against the PostgreSQL database.
    Returns a JWT access token and user profile on success.
    """
    email_clean = str(credentials.email).strip().lower()
    user = db.query(User).filter(User.email == email_clean).first()
    
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account has been deactivated"
        )
        
    access_token = create_access_token(data={"sub": str(user.id), "email": user.email})
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user)
    )

@app.post("/api/auth/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED, summary="User Registration")
def signup(userData: UserSignup, db: Session = Depends(get_db)):
    """
    Register a new user in the PostgreSQL database and immediately return an access token.
    """
    email_clean = str(userData.email).strip().lower()
    existing = db.query(User).filter(User.email == email_clean).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists"
        )
        
    new_user = User(
        email=email_clean,
        hashed_password=hash_password(userData.password),
        full_name=userData.full_name
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token = create_access_token(data={"sub": str(new_user.id), "email": new_user.email})
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(new_user)
    )

@app.get("/api/auth/me", response_model=UserResponse, summary="Get Current Authenticated User")
def get_current_user(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """
    Validate the incoming JWT token and return the currently authenticated user's profile.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header"
        )
        
    token = authorization.split(" ", 1)[1]
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
        
    user_id = int(payload["sub"])
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
        
    return UserResponse.model_validate(user)

@app.get("/api/health", summary="Health Check")
def health_check(db: Session = Depends(get_db)):
    """
    Check API and PostgreSQL connectivity.
    """
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
        
    return {
        "status": "online",
        "database": db_status,
        "version": settings.VERSION
    }

# ---------------------------------------------------------------------------
# Serve Frontend Static Assets
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

