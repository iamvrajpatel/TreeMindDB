"""Coordinates reply delivery with durable, observable memory-tree updates."""
from __future__ import annotations
from collections.abc import Iterator
from app.domain.models import SearchMode

class ChatMemoryService:
    def __init__(self, collection_id, retrieval, chat, ingestion, memory_agent, labeler, repository):
        self.collection_id,self.retrieval,self.chat,self.ingestion,self.memory_agent,self.labeler,self.repository=collection_id,retrieval,chat,ingestion,memory_agent,labeler,repository

    def _answer(self, message: str, top_k: int):
        return self.chat.answer(self.collection_id,message,SearchMode.LAYA,top_k,2)

    def _save_memory(self, message: str, answer: str, job_id: str) -> tuple[list[dict],list[str]]:
        self.repository.update_memory_job(job_id,"extracting","Extracting durable memory notes")
        notes=self.memory_agent.extract(message,answer)
        if not notes:
            self.repository.update_memory_job(job_id,"complete","No durable memory notes found",note_count=0)
            return [],[]
        self.repository.update_memory_job(job_id,"indexing",f"Indexing {len(notes)} memory notes",note_count=len(notes))
        chunk_ids=self.ingestion.ingest_memory_notes(self.collection_id,notes)
        self.repository.save_memory_notes(self.collection_id,notes,chunk_ids)
        self.repository.update_memory_job(job_id,"rebuilding","Rebuilding the semantic memory tree",note_count=len(notes))
        changed_nodes=self.relabel_tree()
        self.repository.update_memory_job(job_id,"complete",f"Saved {len(notes)} memory notes",changed_nodes,len(notes))
        return notes,changed_nodes

    def stream(self, message: str, top_k: int = 5) -> Iterator[dict]:
        job_id=self.repository.create_memory_job(self.collection_id)
        yield {"event":"retrieval_started","job_id":job_id,"status":"Retrieving related memory"}
        response=self._answer(message,top_k); trace=response["retrieval"]
        selected=[node_id for step in trace.steps for node_id in step.selected]
        yield {"event":"laya_route_selected","job_id":job_id,"selected_node_ids":selected,"fallback":trace.fallback,"fallback_mode":"exact" if trace.fallback else None,"steps":[step.__dict__ for step in trace.steps]}
        yield {"event":"answer_ready","job_id":job_id,"answer":response["answer"],"generation_status":response["generation_status"],"citations":response["citations"]}
        if response["generation_status"] != "generated":
            self.repository.update_memory_job(job_id,"failed","Reply generation is unavailable")
            yield {"event":"memory_failed","job_id":job_id,"status":"Reply generation is unavailable; no memory was saved."}
            return
        try:
            yield {"event":"memory_extracting","job_id":job_id,"status":"Extracting concise memory notes"}
            self.repository.update_memory_job(job_id,"extracting","Extracting durable memory notes")
            notes=self.memory_agent.extract(message,response["answer"])
            if not notes:
                self.repository.update_memory_job(job_id,"complete","No durable memory notes found",note_count=0)
                yield {"event":"memory_complete","job_id":job_id,"status":"No durable memory found in this exchange.","saved_notes":[],"changed_node_ids":[]}
                return
            self.repository.update_memory_job(job_id,"indexing",f"Indexing {len(notes)} memory notes",note_count=len(notes))
            yield {"event":"memory_indexing","job_id":job_id,"status":f"Indexing {len(notes)} memory notes","saved_notes":notes}
            chunk_ids=self.ingestion.ingest_memory_notes(self.collection_id,notes)
            self.repository.save_memory_notes(self.collection_id,notes,chunk_ids)
            self.repository.update_memory_job(job_id,"rebuilding","Rebuilding the semantic memory tree",note_count=len(notes))
            yield {"event":"tree_rebuilding","job_id":job_id,"status":"Rebuilding and labeling the semantic tree"}
            changed_nodes=self.relabel_tree()
            self.repository.update_memory_job(job_id,"complete",f"Saved {len(notes)} memory notes",changed_nodes,len(notes))
            yield {"event":"memory_complete","job_id":job_id,"status":"Memory tree is synchronized","saved_notes":notes,"changed_node_ids":changed_nodes}
        except Exception as exc:
            self.repository.update_memory_job(job_id,"failed",f"Memory update failed: {exc.__class__.__name__}")
            yield {"event":"memory_failed","job_id":job_id,"status":"Memory update failed; the reply was not discarded."}

    def send(self, message: str, top_k: int = 5):
        """Synchronous compatibility entry point used by unit tests and non-stream callers."""
        events=list(self.stream(message,top_k)); answer=next((item for item in events if item["event"]=="answer_ready"),{})
        complete=next((item for item in reversed(events) if item["event"] in {"memory_complete","memory_failed"}),{})
        route=next((item for item in events if item["event"]=="laya_route_selected"),{})
        return {"answer":answer.get("answer",""),"generation_status":answer.get("generation_status","unavailable"),"memory_status":"saved" if complete.get("event")=="memory_complete" and complete.get("saved_notes") else "not_saved","saved_notes":complete.get("saved_notes",[]),"citations":answer.get("citations",[]),"trace_id":"","selected_node_ids":route.get("selected_node_ids",[]),"trace":{"steps":route.get("steps",[]),"fallback":route.get("fallback",False),"fallback_mode":route.get("fallback_mode")}}

    def relabel_tree(self) -> list[str]:
        nodes=self.repository.get_nodes(self.collection_id)
        memberships={node.id:self.repository.leaf_vector_ids([node.id]) for node in nodes if not node.child_ids}
        labeled=self.labeler.apply(nodes,memberships,self.repository)
        self.repository.replace_tree(self.collection_id,labeled,memberships)
        return [node.id for node in labeled]
