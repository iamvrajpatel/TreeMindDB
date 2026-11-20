"""Process configuration loaded consistently from the project root."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=False)

def _truthy(value: str | None) -> bool: return (value or "").strip().lower() in {"1","true","yes","on"}

@dataclass(frozen=True)
class Settings:
    data_dir: Path
    cors_origins: list[str]
    ollama_base_url: str
    ollama_model: str | None
    ollama_embedding_model: str | None
    laya_device: str | None
    laya_preload: bool
    context_window_tokens: int
    context_retain_tokens: int

    @classmethod
    def load(cls) -> "Settings":
        configured=Path(os.getenv("TREEMIND_DATA_DIR","data"))
        data_dir=configured if configured.is_absolute() else PROJECT_ROOT / configured
        return cls(data_dir,[value.strip() for value in os.getenv("CORS_ORIGINS","http://localhost:3000").split(",") if value.strip()],os.getenv("OLLAMA_BASE_URL","http://localhost:11434"),os.getenv("OLLAMA_MODEL") or None,os.getenv("OLLAMA_EMBEDDING_MODEL") or None,os.getenv("LAYA_DEVICE") or None,_truthy(os.getenv("LAYA_PRELOAD")),int(os.getenv("CONTEXT_WINDOW_TOKENS","100000")),int(os.getenv("CONTEXT_RETAIN_TOKENS","10000")))
