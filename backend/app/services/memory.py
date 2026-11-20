"""Focused agents for durable chat memory and human-readable tree labels."""
from __future__ import annotations
import json
from dataclasses import replace
from app.rag.generation import UnavailableGenerationProvider

class MemoryAgent:
    def __init__(self, generation): self.generation=generation
    @property
    def available(self): return not isinstance(self.generation,UnavailableGenerationProvider)
    def extract(self, user_message: str, assistant_reply: str) -> list[dict]:
        if not self.available: return []
        prompt=("Extract only durable memory from this exchange. Return JSON only: an array of objects with "
                "kind (fact|preference|decision|constraint|project), topic (short string), and content (one concise bullet). "
                "Exclude greetings, questions with no answer, repetition, and transient chatter.\n\n"
                f"User: {user_message}\nAssistant: {assistant_reply}")
        try: raw=self.generation.generate(prompt); data=json.loads(self._json_array(raw))
        except Exception: return []
        valid=[]
        for item in data if isinstance(data,list) else []:
            if isinstance(item,dict) and item.get("kind") in {"fact","preference","decision","constraint","project"} and isinstance(item.get("topic"),str) and isinstance(item.get("content"),str):
                valid.append({"kind":item["kind"],"topic":item["topic"][:80],"content":item["content"][:600]})
        return valid[:8]
    @staticmethod
    def _json_array(value: str) -> str:
        start,end=value.find("["),value.rfind("]")
        if start<0 or end<start: raise ValueError("Memory agent did not return an array")
        return value[start:end+1]

class ClusterLabeler:
    def __init__(self, generation): self.generation=generation
    def label(self, notes: list[dict], depth: int) -> tuple[str,str]:
        examples="; ".join(f"{n['topic']}: {n['content']}" for n in notes[:5])
        if not examples: return ("Empty memory", "No durable memory notes in this node.")
        if isinstance(self.generation,UnavailableGenerationProvider):
            topic=notes[0]["topic"].strip() or "Chat memory"; return (topic[:60],f"Memory about {topic} ({len(notes)} notes).")
        prompt=("Create a JSON object with `label` (max 6 words) and `summary` (one sentence) that describes these related chat memory notes. "
                f"Tree depth: {depth}. Notes: {examples}")
        try:
            parsed=json.loads(self.generation.generate(prompt)); return (str(parsed["label"])[:60],str(parsed["summary"])[:280])
        except Exception:
            return (notes[0]["topic"][:60],f"Memory about {notes[0]['topic']} ({len(notes)} notes).")
    def apply(self, nodes, memberships, repository):
        relabeled=[]
        descendants={node.id:set(memberships.get(node.id,[])) for node in nodes}
        for node in sorted(nodes,key=lambda item:item.depth,reverse=True):
            for child in node.child_ids: descendants[node.id].update(descendants.get(child,set()))
            notes=repository.notes_for_vector_ids(list(descendants[node.id])); label,summary=self.label(notes,node.depth)
            relabeled.append(replace(node,label=label,summary=summary))
        return relabeled
