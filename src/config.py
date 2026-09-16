import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

class Settings:
    PROJECT_NAME: str = "Patient-Record RAG Prototype"
    VERSION: str = "1.0.0"
    
    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_PATH: Path = BASE_DIR / os.getenv("DATA_PATH", "data/synthetic_patients.json")
    
    # On Vercel / serverless, writeable directory is /tmp
    if os.getenv("VERCEL"):
        CHROMA_DB_DIR: Path = Path("/tmp/chroma_db")
    else:
        CHROMA_DB_DIR: Path = BASE_DIR / os.getenv("CHROMA_DB_DIR", "chroma_db")
        
    STATIC_DIR: Path = BASE_DIR / "static"
    
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_API_KEY: str = GEMINI_API_KEY
    
    # Model Configuration
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

settings = Settings()
