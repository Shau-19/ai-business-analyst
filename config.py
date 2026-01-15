# config.py
"""
Configuration for AI Business Analyst
"""
'''
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings"""
    
    # ============== API KEYS ==============
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # ============== LLM SETTINGS ==============
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    TEMPERATURE: float = 0.1  # Low for factual SQL generation
    MAX_TOKENS: int = 2000
    
    # ============== DATABASE ==============
    DB_PATH: str = "./data/business.db"
    
    # ============== SERVER ==============
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # ============== LOGGING ==============
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/analyst.log"
    
    # ============== SUPPORTED LANGUAGES ==============
    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "ru", "zh-cn", 
        "ja", "ko", "ar", "hi", "bn", "ur", "pa"
    ]
    
    def __init__(self):
        """Initialize and validate configuration"""
        
        # Validate API key
        if not self.GROQ_API_KEY:
            raise ValueError(
                "❌ GROQ_API_KEY not found!\n"
                "Please set it in .env file:\n"
                "GROQ_API_KEY=your_key_here\n"
                "Get your key from: https://console.groq.com/keys"
            )
        
        # Create directories
        directories = [
            "./data",
            "./logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        print("✅ Configuration loaded successfully")


# Create global settings instance
settings = Settings()


'''


# config.py
"""
Configuration Module - Complete Final Version
"""
import os
from dotenv import load_dotenv
from pathlib import Path
from dataclasses import dataclass

# Load environment variables
load_dotenv()

# ============================================================================
# SETTINGS CLASS (for backward compatibility with 'from config import settings')
# ============================================================================

@dataclass
class Settings:
    """Application settings"""
    
    # Database
    DB_TYPE: str = os.getenv("DB_TYPE", "sqlite")
    DB_PATH: str = os.getenv("DB_PATH", "./data/business.db")
    
    # LLM
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
    
    # RAG
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "faiss")
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_stores")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    SEARCH_K: int = int(os.getenv("SEARCH_K", "8"))
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/app.log")
    
    # Session
    SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "3600"))
    
    # CSV Hybrid (NEW)
    CSV_PROCESSING_MODE: str = os.getenv("CSV_PROCESSING_MODE", "hybrid")
    CSV_AUTO_IMPORT_SQL: bool = os.getenv("CSV_AUTO_IMPORT_SQL", "True").lower() == "true"
    CSV_AUTO_INDEX_RAG: bool = os.getenv("CSV_AUTO_INDEX_RAG", "True").lower() == "true"
    CSV_TABLE_PREFIX: str = os.getenv("CSV_TABLE_PREFIX", "upload_")
    
    # Feature Flags
    ENABLE_MCP: bool = os.getenv("ENABLE_MCP", "True").lower() == "true"
    ENABLE_A2A: bool = os.getenv("ENABLE_A2A", "True").lower() == "true"
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "False").lower() == "true"
    ENABLE_HYBRID_CSV: bool = os.getenv("ENABLE_HYBRID_CSV", "True").lower() == "true"


# Create global settings instance
settings = Settings()

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DB_TYPE = settings.DB_TYPE
DB_PATH = settings.DB_PATH

POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "business_db"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "password")
}

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "database": os.getenv("MYSQL_DB", "business_db"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "password")
}

def get_db_config():
    """Get active database configuration"""
    if DB_TYPE == "sqlite":
        return {"type": "sqlite", "path": DB_PATH}
    elif DB_TYPE == "postgresql":
        return {"type": "postgresql", **POSTGRES_CONFIG}
    elif DB_TYPE == "mysql":
        return {"type": "mysql", **MYSQL_CONFIG}
    else:
        raise ValueError(f"Unsupported DB_TYPE: {DB_TYPE}")

# ============================================================================
# EXPORTED CONSTANTS (for direct import)
# ============================================================================

GROQ_API_KEY = settings.GROQ_API_KEY
LLM_MODEL = settings.LLM_MODEL
TEMPERATURE = settings.TEMPERATURE
MAX_TOKENS = settings.MAX_TOKENS

EMBEDDING_MODEL = settings.EMBEDDING_MODEL
VECTOR_STORE_TYPE = settings.VECTOR_STORE_TYPE
VECTOR_STORE_PATH = settings.VECTOR_STORE_PATH
CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
SEARCH_K = settings.SEARCH_K

HOST = settings.HOST
PORT = settings.PORT
DEBUG = settings.DEBUG

LOG_LEVEL = settings.LOG_LEVEL
LOG_FILE = settings.LOG_FILE

SESSION_TIMEOUT = settings.SESSION_TIMEOUT

CSV_PROCESSING_MODE = settings.CSV_PROCESSING_MODE
CSV_AUTO_IMPORT_SQL = settings.CSV_AUTO_IMPORT_SQL
CSV_AUTO_INDEX_RAG = settings.CSV_AUTO_INDEX_RAG
CSV_TABLE_PREFIX = settings.CSV_TABLE_PREFIX

ENABLE_MCP = settings.ENABLE_MCP
ENABLE_A2A = settings.ENABLE_A2A
ENABLE_CACHING = settings.ENABLE_CACHING
ENABLE_HYBRID_CSV = settings.ENABLE_HYBRID_CSV

# ============================================================================
# PATHS SETUP
# ============================================================================

Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
if DB_TYPE == "sqlite":
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# ============================================================================
# RUNTIME INFO
# ============================================================================

def print_config():
    """Print current configuration"""
    print("="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Database Type:     {DB_TYPE}")
    if DB_TYPE == "sqlite":
        print(f"Database Path:     {DB_PATH}")
    else:
        config = get_db_config()
        print(f"Database Host:     {config['host']}:{config['port']}")
        print(f"Database Name:     {config['database']}")
    print(f"LLM Model:         {LLM_MODEL}")
    print(f"Embedding Model:   {EMBEDDING_MODEL}")
    print(f"Vector Store:      {VECTOR_STORE_TYPE}")
    print(f"CSV Hybrid:        {ENABLE_HYBRID_CSV}")
    print(f"Server:            {HOST}:{PORT}")
    print("="*60)


if __name__ == "__main__":
    print_config()