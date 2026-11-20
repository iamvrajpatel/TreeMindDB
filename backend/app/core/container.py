from __future__ import annotations
from pathlib import Path
from app.storage.repository import SqliteTreeRepository
from app.storage.vector_store import BinaryVectorStore
from app.embeddings.providers import DeterministicEmbeddingProvider, OllamaEmbeddingProvider
from app.ingestion.extractors import SafeTextExtractor
from app.services.tree_builder import HierarchicalKMeansBuilder
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.routing.routers import CentroidRouter,LayaDecisionRouter
from app.core.settings import Settings
from app.rag.generation import UnavailableGenerationProvider, OllamaGenerationProvider, OpenAICompatibleGenerationProvider
from app.services.chat import ChatService
from app.services.chat_memory import ChatMemoryService
from app.services.memory import MemoryAgent, ClusterLabeler
from app.services.personal_knowledge import PersonalKnowledgeService
from app.services.personal_assistant import PersonalAssistantService
from app.services.context_memory import ActiveContextService
from app.services.context_assistant import ContextFirstAssistant
import os
from dataclasses import replace

class Container:
    def __init__(self, data_dir:str | None = None, settings: Settings | None = None):
        self.settings=settings or Settings.load(); path=Path(data_dir) if data_dir else self.settings.data_dir; path.mkdir(parents=True,exist_ok=True)
        self.repository=SqliteTreeRepository(str(path/"treemind.db"))
        self.embedding_status={"mode":"demo","reason":"No OLLAMA_EMBEDDING_MODEL configured"}
        # Explicit data directories are used by the deterministic demo/test
        # adapter. The normal application runtime (no override) uses Ollama.
        if self.settings.ollama_embedding_model and data_dir is None:
            try:
                self.embeddings=OllamaEmbeddingProvider(self.settings.ollama_embedding_model,self.settings.ollama_base_url)
                self.embedding_status={"mode":"semantic","reason":None}
            except Exception as exc:
                self.embeddings=DeterministicEmbeddingProvider()
                self.embedding_status={"mode":"degraded","reason":f"Ollama embeddings unavailable: {exc.__class__.__name__}"}
        else:self.embeddings=DeterministicEmbeddingProvider()
        self.vectors=BinaryVectorStore(str(path/"vectors")); self.laya=LayaDecisionRouter(self.settings)
        self.ingestion=IngestionService(self.repository,self.vectors,self.embeddings,SafeTextExtractor(),HierarchicalKMeansBuilder())
        self.retrieval=RetrievalService(self.repository,self.vectors,self.embeddings,CentroidRouter(),self.laya)
        if self.settings.ollama_model:
            self.generation=OllamaGenerationProvider(self.settings.ollama_model,self.settings.ollama_base_url)
        elif os.getenv("OPENAI_COMPATIBLE_MODEL") and os.getenv("OPENAI_COMPATIBLE_BASE_URL") and os.getenv("OPENAI_COMPATIBLE_API_KEY"):
            self.generation=OpenAICompatibleGenerationProvider(os.environ["OPENAI_COMPATIBLE_MODEL"],os.environ["OPENAI_COMPATIBLE_BASE_URL"],os.environ["OPENAI_COMPATIBLE_API_KEY"])
        else: self.generation=UnavailableGenerationProvider()
        self.chat=ChatService(self.retrieval,self.generation)
        self.chat_memory_collection=self._embedding_collection("chat-memory",branching_factor=3,max_leaf_size=4,minimum_split_size=2)
        # Chat memory should form inspectable branches after a handful of durable notes, not 128.
        if self.chat_memory_collection.max_leaf_size != 4 or self.chat_memory_collection.branching_factor != 3 or self.chat_memory_collection.minimum_split_size != 2:
            self.chat_memory_collection=self.repository.update_collection_config(replace(self.chat_memory_collection,branching_factor=3,max_leaf_size=4,minimum_split_size=2))
        # Resolve compatibility for every shared vector segment before services
        # capture the embedding provider.
        self.personal_collection=self._embedding_collection("personal-knowledge",branching_factor=3,max_leaf_size=4,minimum_split_size=2)
        self.context_collection=self._embedding_collection("active-context")
        self.ingestion.embedding=self.embeddings
        self.retrieval.embedding=self.embeddings
        self.memory_agent=MemoryAgent(self.generation); self.labeler=ClusterLabeler(self.generation)
        self.chat_memory=ChatMemoryService(self.chat_memory_collection.id,self.retrieval,self.chat,self.ingestion,self.memory_agent,self.labeler,self.repository)
        self.personal_knowledge=PersonalKnowledgeService(self.personal_collection.id,self.repository,self.vectors,self.embeddings,self.laya,self.generation,self.labeler)
        self.personal_assistant=PersonalAssistantService(self.personal_knowledge)
        self.active_context=ActiveContextService(self.context_collection.id,self.repository,self.vectors,self.embeddings,self.generation,self.settings.context_window_tokens,self.settings.context_retain_tokens)
        self.context_assistant=ContextFirstAssistant(self.active_context,self.personal_knowledge,self.generation)

    def _embedding_collection(self, name: str, **config):
        collection=self.repository.get_or_create_collection(name,embedding_model=self.embeddings.model_id,embedding_dimension=self.embeddings.dimension,**config)
        if collection.embedding_model == self.embeddings.model_id and collection.embedding_dimension == self.embeddings.dimension:
            return collection
        if self.repository.stats(collection.id)["vectors"] == 0:
            return self.repository.update_collection_config(replace(collection,embedding_model=self.embeddings.model_id,embedding_dimension=self.embeddings.dimension))
        # Existing MVP data used deterministic vectors. Keep it readable and make
        # the required rebuild explicit rather than corrupting the vector segment.
        if collection.embedding_model == "deterministic-hash-v1":
            self.embeddings=DeterministicEmbeddingProvider(collection.embedding_dimension)
            self.embedding_status={"mode":"rebuild-required","reason":"Existing deterministic vectors require re-embedding before semantic mode can be enabled"}
            return collection
        raise RuntimeError(f"Collection '{name}' uses {collection.embedding_model}; re-embed it before changing models")
