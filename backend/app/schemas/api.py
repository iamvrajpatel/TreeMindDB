from __future__ import annotations
from pydantic import BaseModel, Field
from app.domain.models import SearchMode

class CollectionCreate(BaseModel):
    name: str = Field(min_length=1,max_length=100)
    embedding_dimension: int | None = Field(default=None,ge=8,le=4096)
    branching_factor: int = Field(default=4,ge=2,le=32)
    max_leaf_size: int = Field(default=128,ge=2,le=10000)
    max_tree_depth: int = Field(default=8,ge=1,le=32)
    minimum_split_size: int = Field(default=8,ge=2,le=10000)

class TextIngest(BaseModel):
    name: str = Field(default="pasted-text.txt",max_length=255)
    text: str = Field(min_length=1,max_length=5_000_000)

class SearchRequest(BaseModel):
    query: str = Field(min_length=1,max_length=10000)
    mode: SearchMode = SearchMode.HYBRID
    top_k: int = Field(default=5,ge=1,le=100)
    beam_width: int = Field(default=2,ge=1,le=16)

class ChatRequest(SearchRequest):
    conversation_id: str | None = None

class ChatMemoryMessage(BaseModel):
    message: str = Field(min_length=1,max_length=200000)
    top_k: int = Field(default=5,ge=1,le=20)
    conversation_id: str | None = None
