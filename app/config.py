from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Database settings
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    # Application settings
    APP_NAME: str
    ENVIRONMENT: str
    OPENAI_API_KEY: str
    GEMINI_API_KEY: str

    # File paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_PDF_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    MEDIA_DIR: Path = DATA_DIR / "media"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.DATA_DIR, self.RAW_PDF_DIR, self.PROCESSED_DIR, self.MEDIA_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

# Create settings instance
settings = Settings() 