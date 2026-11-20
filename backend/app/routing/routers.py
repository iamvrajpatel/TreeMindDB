from __future__ import annotations
import time
from app.core.settings import Settings
from app.domain.models import TreeNode

class CentroidRouter:
    def rank_children(self, query: str, parent: TreeNode, children: list[TreeNode], query_vector=None):
        if query_vector is None: raise ValueError("Centroid routing requires query_vector")
        return sorted(((c.id,float(query_vector @ c.centroid)) for c in children),key=lambda item:item[1],reverse=True)

class LayaDecisionRouter:
    """Thin adapter; unsupported/malformed SDK output raises for caller-controlled fallback."""
    def __init__(self, settings: Settings | None = None):
        self.settings=settings or Settings.load(); self.state="starting"
        try:
            from laya import Router
            self._router=Router(device=self.settings.laya_device) if self.settings.laya_device else Router()
            if self.settings.laya_preload: self._router.preload(["english","multilingual"])
            self.available=True; self.reason=None; self.state="ready"
        except Exception as exc:
            self._router=None; self.available=False; self.reason=f"Laya unavailable: {exc.__class__.__name__}"; self.state="error"
    def rank_children(self, query: str, parent: TreeNode, children: list[TreeNode]):
        if not self.available: raise RuntimeError(self.reason)
        criteria={c.id:(c.label or c.id)+": "+c.summary for c in children}
        question={"branch":{"type":"choice","instructions":"Select the tree branch most relevant to retrieving the user query.","criteria":criteria}}
        started=time.perf_counter(); result=self._router.predict({"query":query,"node":parent.summary},question)
        answer=result.get("answers",{}).get("branch",{}); choice=answer.get("choice")
        if choice not in criteria: raise ValueError("Laya returned an invalid child identifier")
        probabilities=answer.get("probabilities") or answer.get("scores")
        if isinstance(probabilities,dict): return sorted(((key,float(probabilities.get(key,0.0))) for key in criteria),key=lambda x:x[1],reverse=True)
        # The SDK supplied only a selected choice; preserve it without inventing scores.
        return [(choice, 1.0)] + [(key, 0.0) for key in criteria if key != choice]
