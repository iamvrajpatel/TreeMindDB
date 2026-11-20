from __future__ import annotations
from collections.abc import Iterator

class PersonalAssistantService:
    def __init__(self, knowledge): self.knowledge=knowledge
    def stream(self, message: str, top_k: int = 5) -> Iterator[dict]:
        job_id=self.knowledge.repository.create_memory_job(self.knowledge.collection_id)
        yield {"event":"retrieval_started","job_id":job_id,"status":"Searching your personal knowledge"}
        answer,retrieval=self.knowledge.answer(message,top_k)
        yield {"event":"laya_route_selected","job_id":job_id,"selected_node_ids":retrieval.selected_node_ids,"steps":retrieval.steps,"fallback":retrieval.fallback,"fallback_mode":"exact" if retrieval.fallback else None}
        yield {"event":"answer_ready","job_id":job_id,"answer":answer,"citations":[hit.item_id for hit in retrieval.hits]}
        try:
            self.knowledge.repository.update_memory_job(job_id,"curating","Finding durable personal knowledge")
            yield {"event":"knowledge_planning","job_id":job_id,"status":"Checking for information to add or update"}
            items=self.knowledge.curator.extract(message)
            if not items:
                self.knowledge.repository.update_memory_job(job_id,"complete","No new durable information",note_count=0)
                yield {"event":"tree_complete","job_id":job_id,"status":"No personal knowledge change was needed.","created_node_ids":[],"added_item_ids":[],"pending_update_ids":[]}
                return
            self.knowledge.repository.update_memory_job(job_id,"indexing",f"Applying {len(items)} personal knowledge updates",note_count=len(items))
            yield {"event":"knowledge_indexing","job_id":job_id,"status":f"Adding {len(items)} durable knowledge items"}
            nodes,added,pending=self.knowledge.persist_items(items)
            node_ids=[node.id for node in nodes]
            self.knowledge.repository.update_memory_job(job_id,"complete",f"Personal knowledge tree synchronized",node_ids,len(added))
            for node in nodes: yield {"event":"branch_created","job_id":job_id,"node_id":node.id,"label":node.label}
            for item_id in added: yield {"event":"fact_added","job_id":job_id,"item_id":item_id}
            for update_id in pending: yield {"event":"fact_update_pending","job_id":job_id,"update_id":update_id,"status":"A conflicting current fact needs your confirmation."}
            yield {"event":"tree_complete","job_id":job_id,"status":"Personal knowledge tree synchronized","created_node_ids":node_ids,"added_item_ids":added,"pending_update_ids":pending}
        except Exception as exc:
            self.knowledge.repository.update_memory_job(job_id,"failed",f"Knowledge update failed: {exc.__class__.__name__}")
            yield {"event":"tree_failed","job_id":job_id,"status":"The reply was sent, but the knowledge tree update failed."}
