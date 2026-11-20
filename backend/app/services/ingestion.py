from __future__ import annotations
import hashlib
from app.ingestion.extractors import chunk_pages

class IngestionService:
    def __init__(self, repository, vector_store, embedding_provider, extractor, tree_builder):
        self.repository,self.vector_store,self.embedding,self.extractor,self.builder=repository,vector_store,embedding_provider,extractor,tree_builder
    def ingest_bytes(self, collection_id:str, filename:str, content:bytes):
        config=self.repository.get_collection(collection_id)
        if not config: raise KeyError("Collection not found")
        if len(content)>25*1024*1024: raise ValueError("Upload exceeds 25 MB limit")
        pages=self.extractor.extract(content,filename); chunks=chunk_pages(pages)
        if not chunks: raise ValueError("The document did not contain indexable text")
        vectors=self.embedding.embed([chunk[0] for chunk in chunks])
        if vectors.shape[1]!=config.embedding_dimension: raise ValueError("Embedding dimension is incompatible with collection")
        offsets=self.vector_store.append(collection_id,vectors)
        doc_id,chunk_ids=self.repository.insert_document(collection_id,filename,hashlib.sha256(content).hexdigest(),chunks,offsets)
        self.rebuild(collection_id)
        return {"document_id":doc_id,"chunk_ids":chunk_ids,"vectors":len(chunks),"bytes":len(content)}
    def rebuild(self, collection_id:str):
        config=self.repository.get_collection(collection_id); records=self.repository.active_vectors(collection_id)
        vectors=self.vector_store.read(collection_id,[r['offset'] for r in records],config.embedding_dimension)
        built=self.builder.build(config,[r['vector_id'] for r in records],vectors,[r['document_id'] for r in records])
        self.repository.replace_tree(collection_id,built.nodes,built.memberships)
        return {"nodes":len(built.nodes),"vectors":len(records)}
    def delete_document(self, collection_id: str, document_id: str):
        if not self.repository.delete_document(collection_id,document_id): raise KeyError("Document not found")
        return self.rebuild(collection_id)
    def ingest_memory_notes(self, collection_id: str, notes: list[dict]) -> list[str]:
        """Store one compact vector per memory note, never raw conversation text."""
        config=self.repository.get_collection(collection_id)
        records=[(f"[{note['kind']}] {note['topic']}: {note['content']}",None,len(note['content'].split())) for note in notes]
        vectors=self.embedding.embed([record[0] for record in records])
        if vectors.shape[1] != config.embedding_dimension: raise ValueError("Embedding dimension is incompatible with collection")
        offsets=self.vector_store.append(collection_id,vectors)
        _,chunk_ids=self.repository.insert_document(collection_id,"chat-memory-notes.md","chat-memory",records,offsets)
        self.rebuild(collection_id)
        return chunk_ids
