import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    model_fast: str = os.getenv("MODEL_FAST", "gemini-2.5-flash-lite")
    model_strong: str = os.getenv("MODEL_STRONG", "gemini-2.5-flash")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "10"))
    max_workers: int = int(os.getenv("MAX_WORKERS", "4"))
    api_key: str | None = os.getenv("GEMINI_API_KEY")

SETTINGS = Settings()
