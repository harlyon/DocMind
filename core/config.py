from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_name: str = "DocMind"
    app_env: str = "development"
    secret_key: str = "change-me-in-production"

    # Google Gemini
    # Get a free key at https://aistudio.google.com/app/apikey
    google_api_key: str = ""
    llm_model: str = "gemini-2.5-flash"

    # Embedding model — still local via sentence-transformers (free, no API key needed)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/docmind.db"

    # File storage
    upload_dir: str = "./data/uploads"

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 50
    # Gemini 2.5 Flash has a 1M token context window — use more chunks than
    # you would with a smaller model for richer, more accurate answers
    top_k_retrieval: int = 6

    # CORS
    allowed_origins: str = "http://localhost:3000"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()