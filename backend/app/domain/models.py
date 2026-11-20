"""Framework-independent TreeMind domain types."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

class SearchMode(StrEnum):
    EXACT = "exact"
    CENTROID = "centroid"
    LAYA = "laya"
    HYBRID = "hybrid"

class NodeType(StrEnum):
    ROOT = "root"
    INTERNAL = "internal"
    LEAF = "leaf"

@dataclass(frozen=True)
class CollectionConfig:
    id: str
    name: str
    embedding_model: str = "deterministic-hash-v1"
    embedding_dimension: int = 96
    branching_factor: int = 4
    max_leaf_size: int = 128
    max_tree_depth: int = 8
    minimum_split_size: int = 8

@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    position: int
    token_count: int
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TreeNode:
    id: str
    collection_id: str
    parent_id: str | None
    node_type: NodeType
    depth: int
    child_ids: tuple[str, ...]
    centroid: tuple[float, ...]
    vector_count: int
    document_count: int
    radius: float
    summary: str
    label: str | None

@dataclass(frozen=True)
class RoutingStep:
    node_id: str
    depth: int
    candidates: list[dict[str, Any]]
    selected: list[str]
    strategy: str
    fallback: bool = False
    fallback_reason: str | None = None
    latency_ms: float = 0.0

@dataclass(frozen=True)
class SearchHit:
    vector_id: str
    chunk_id: str
    document_id: str
    score: float
    text: str
    source_name: str

@dataclass(frozen=True)
class SearchResult:
    trace_id: str
    mode: SearchMode
    hits: list[SearchHit]
    steps: list[RoutingStep]
    visited_nodes: int
    candidate_vectors: int
    fallback: bool
    total_latency_ms: float
