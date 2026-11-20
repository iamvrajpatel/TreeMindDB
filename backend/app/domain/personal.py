from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class PersonalNode:
    id: str
    parent_id: str | None
    label: str
    summary: str
    depth: int
    node_type: str
    created_at: str
    updated_at: str

@dataclass(frozen=True)
class PersonalSearchHit:
    item_id: str
    node_id: str
    topic: str
    content: str
    kind: str
    score: float

@dataclass(frozen=True)
class PersonalSearchResult:
    hits: list[PersonalSearchHit]
    selected_node_ids: list[str]
    steps: list[dict[str, Any]]
    fallback: bool
