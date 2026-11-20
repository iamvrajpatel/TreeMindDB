"""Persisted active conversation windows and rollover into semantic knowledge."""
from __future__ import annotations
import hashlib, json, math
import numpy as np
from app.domain.context import ContextHit
from app.rag.generation import UnavailableGenerationProvider

class TokenCounter:
    """Conservative local estimate; configurable headroom protects the model context limit."""
    def count(self, text: str) -> int: return max(1, math.ceil(len(text) / 3.5))

class ContextEvidenceAssessor:
    def __init__(self, generation): self.generation=generation
    def sufficient(self, query: str, evidence: list[ContextHit]) -> bool:
        if not evidence or isinstance(self.generation,UnavailableGenerationProvider): return False
        sample="\n".join(f"[{hit.role}] {hit.text}" for hit in evidence[:8])
        prompt=("Return JSON only: {\"sufficient\": true|false}. Is the supplied active conversation evidence enough to answer the query without outside personal knowledge? "
                "Use true only when it directly supports a useful answer.\n\nQuery: "+query+"\nEvidence:\n"+sample)
        try:return bool(json.loads(self.generation.generate(prompt))["sufficient"])
        except Exception:return False

class ActiveContextService:
    def __init__(self, collection_id, repository, vectors, embedding, generation, window_tokens=100_000, retain_tokens=10_000):
        self.collection_id,self.repository,self.vectors,self.embedding,self.generation=collection_id,repository,vectors,embedding,generation
        self.window_tokens,self.retain_tokens=window_tokens,retain_tokens; self.counter=TokenCounter(); self.assessor=ContextEvidenceAssessor(generation)
    def get_or_create(self, conversation_id: str | None):
        return self.repository.conversation(conversation_id) or self.repository.create_conversation()
    def append(self, conversation_id: str, role: str, text: str):
        count=self.counter.count(text); vector=self.embedding.embed([text]); offset=self.vectors.append(self.collection_id,vector)[0]
        _,chunks=self.repository.insert_document(self.collection_id,"active-context.md",hashlib.sha256((conversation_id+role+text).encode()).hexdigest(),[(text,None,count)],[offset])
        segment_id=self.repository.add_context_segment(conversation_id,chunks[0],role,text,count)
        conversation=self.repository.conversation(conversation_id); self.repository.update_conversation(conversation_id,active_tokens=conversation.active_tokens+count)
        return segment_id,count
    def search(self, conversation_id: str, query: str, top_k=6):
        records=self.repository.active_context_records(); current_latest=self.repository.recent_context_records(conversation_id,1)
        excluded={current_latest[0]["id"]} if current_latest and current_latest[0]["role"]=="user" and current_latest[0]["text"]==query else set()
        records=[record for record in records if record["id"] not in excluded]
        if not records:return [],False
        config=self.repository.get_collection(self.collection_id); q=self.embedding.embed([query])[0]; matrix=self.vectors.read(self.collection_id,[record["offset"] for record in records],config.embedding_dimension); scores=matrix@q
        order=np.argsort(scores)[::-1][:top_k]; hits=[ContextHit(records[i]["id"],records[i]["conversation_id"],records[i]["role"],records[i]["text"],float(scores[i])) for i in order]
        # Recent local turns always accompany semantically retrieved active context.
        recent=self.repository.recent_context_records(conversation_id,4); seen={hit.segment_id for hit in hits}
        for record in recent:
            if record["id"] not in seen:hits.append(ContextHit(record["id"],record["conversation_id"],record["role"],record["text"],1.0));seen.add(record["id"])
        return hits,self.assessor.sufficient(query,hits)
    def needs_rollover(self, conversation_id: str):
        conversation=self.repository.conversation(conversation_id); return bool(conversation and conversation.active_tokens>=self.window_tokens)
    def rollover(self, conversation_id: str, curator, knowledge):
        active=self.repository.active_context_records(conversation_id); total=sum(item["token_count"] for item in active)
        retained=[]; kept=0
        for item in reversed(active):
            if kept+item["token_count"]<=self.retain_tokens:retained.append(item);kept+=item["token_count"]
            else:break
        retained_ids={item["id"] for item in retained}; archived=[item for item in active if item["id"] not in retained_ids]
        # Durable knowledge is written by the explicit checkpointed sync before
        # rollover. Raw archived turns are never copied into the personal tree.
        created,added,pending=[],[],[]
        self.repository.archive_context_segments([item["id"] for item in archived])
        profile=self._profile([item["text"] for item in retained])
        self.repository.update_conversation(conversation_id,active_tokens=sum(item["token_count"] for item in retained),profile_summary=profile)
        return {"archived_tokens":sum(item["token_count"] for item in archived),"retained_tokens":sum(item["token_count"] for item in retained),"profile_summary":profile,"created_nodes":created,"added_item_ids":added,"pending_update_ids":pending}
    def _profile(self, texts: list[str]):
        if not texts:return "Active context reset; knowledge was stored in the personal tree."
        if isinstance(self.generation,UnavailableGenerationProvider):return "Active conversation context retained."
        prompt="Summarize the active conversation context in at most 120 words. Keep only current topic, user goals, constraints, and unresolved questions.\n\n"+"\n".join(texts)
        try:return self.generation.generate(prompt)[:1000]
        except Exception:return "Active conversation context retained."
