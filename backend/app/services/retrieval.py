from __future__ import annotations
import time, uuid
import numpy as np
from app.domain.models import RoutingStep, SearchHit, SearchMode, SearchResult

class RetrievalService:
    def __init__(self, repository, vector_store, embedding_provider, centroid_router, laya_router):
        self.repository,self.vector_store,self.embedding,self.centroid,self.laya=repository,vector_store,embedding_provider,centroid_router,laya_router
    def search(self, collection_id: str, query: str, mode: SearchMode, top_k:int=5, beam_width:int=2) -> SearchResult:
        started=time.perf_counter(); config=self.repository.get_collection(collection_id)
        if not config: raise KeyError("Collection not found")
        q=self.embedding.embed([query])[0]; nodes=self.repository.get_nodes(collection_id); by_id={n.id:n for n in nodes}; root=next((n for n in nodes if n.parent_id is None),None)
        steps=[]; fallback=False
        all_records=self.repository.active_vectors(collection_id)
        candidate_ids=[r['vector_id'] for r in all_records]
        force_exact = mode == SearchMode.LAYA and not getattr(self.laya, "available", True)
        if force_exact:
            fallback=True
            if root:
                steps.append(RoutingStep(root.id,root.depth,[],[],"exact-fallback",True,getattr(self.laya,"reason","Laya unavailable"),0.0))
        if mode != SearchMode.EXACT and root and not force_exact:
            frontier=[root]
            while frontier:
                if all(not node.child_ids for node in frontier):
                    break
                next_frontier=[]
                for parent in frontier:
                    children=[by_id[c] for c in parent.child_ids]
                    if not children: next_frontier.append(parent); continue
                    route_started=time.perf_counter(); reason=None
                    try:
                        if mode==SearchMode.CENTROID: ranked=self.centroid.rank_children(query,parent,children,q); strategy="centroid"
                        elif mode==SearchMode.LAYA: ranked=self.laya.rank_children(query,parent,children); strategy="laya"
                        else:
                            geometric=dict(self.centroid.rank_children(query,parent,children,q)); decision=dict(self.laya.rank_children(query,parent,children)); ranked=sorted(((c.id,0.65*geometric[c.id]+0.35*decision.get(c.id,0.0)) for c in children),key=lambda x:x[1],reverse=True); strategy="hybrid"
                    except Exception as exc:
                        if mode == SearchMode.LAYA:
                            # The simplified chat product never silently changes tree-routing algorithms.
                            fallback=True; candidate_ids=[r['vector_id'] for r in all_records]
                            steps.append(RoutingStep(parent.id,parent.depth,[],[],"exact-fallback",True,str(exc),(time.perf_counter()-route_started)*1000))
                            frontier=[]; break
                        ranked=self.centroid.rank_children(query,parent,children,q); strategy="centroid-fallback"; fallback=True; reason=str(exc)
                    selected=[x[0] for x in ranked[:beam_width]]
                    steps.append(RoutingStep(parent.id,parent.depth,[{"node_id":i,"routing_score":s,"centroid_similarity":float(q @ by_id[i].centroid)} for i,s in ranked],selected,strategy,reason is not None,reason,(time.perf_counter()-route_started)*1000))
                    next_frontier.extend(by_id[n] for n in selected)
                if not frontier: break
                frontier=next_frontier
            if not fallback:
                leaves=[n.id for n in frontier if not n.child_ids]
                candidate_ids=self.repository.leaf_vector_ids(leaves)
        records=self.repository.vector_records(candidate_ids)
        offsets=[r['offset'] for r in records]; vectors=self.vector_store.read(collection_id,offsets,config.embedding_dimension)
        scores=vectors@q if len(vectors) else np.array([]); ordering=np.argsort(scores)[::-1][:top_k]
        hits=[SearchHit(records[i]['vector_id'],records[i]['chunk_id'],records[i]['document_id'],float(scores[i]),records[i]['text'],records[i]['name']) for i in ordering]
        trace_id=str(uuid.uuid4()); result=SearchResult(trace_id,mode,hits,steps,len({s.node_id for s in steps}),len(records),fallback,(time.perf_counter()-started)*1000)
        self.repository.save_trace(collection_id,trace_id,{"trace_id":trace_id,"mode":mode,"steps":[s.__dict__ for s in steps],"fallback":fallback,"hits":[h.__dict__ for h in hits]})
        return result
