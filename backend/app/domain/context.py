from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    active_tokens: int
    profile_summary: str
    created_at: str
    updated_at: str

@dataclass(frozen=True)
class ContextHit:
    segment_id: str
    conversation_id: str
    role: str
    text: str
    score: float
