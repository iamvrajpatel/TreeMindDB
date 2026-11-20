from __future__ import annotations
import uuid
from dataclasses import dataclass
import numpy as np
from app.domain.models import CollectionConfig, NodeType, TreeNode

@dataclass
class BuiltTree:
    nodes: list[TreeNode]
    memberships: dict[str, list[str]]

class HierarchicalKMeansBuilder:
    """Deterministic cosine tree builder; all vectors are normalized before clustering."""
    def build(self, config: CollectionConfig, vector_ids: list[str], vectors: np.ndarray, document_ids: list[str]) -> BuiltTree:
        if not len(vector_ids): return BuiltTree([], {})
        norms=np.linalg.norm(vectors,axis=1,keepdims=True); vectors=vectors/np.maximum(norms,1e-12)
        nodes: list[TreeNode]=[]; memberships: dict[str,list[str]]={}
        def create(indices: np.ndarray, parent: str|None, depth: int, root: bool=False) -> str:
            node_id=str(uuid.uuid4()); subset=vectors[indices]; centroid=subset.mean(axis=0); centroid/=max(np.linalg.norm(centroid),1e-12)
            radius=float(np.max(1-subset@centroid)) if len(subset) else 0.0
            is_leaf=len(indices)<=config.max_leaf_size or depth>=config.max_tree_depth or len(indices)<config.minimum_split_size
            children: list[str]=[]
            if not is_leaf:
                groups=self._cluster(subset, config.branching_factor)
                if len(groups)<2: is_leaf=True
                else: children=[create(indices[group],node_id,depth+1) for group in groups]
            kind=NodeType.ROOT if root else (NodeType.LEAF if is_leaf else NodeType.INTERNAL)
            docs={document_ids[i] for i in indices}
            node=TreeNode(node_id,config.id,parent,kind,depth,tuple(children),tuple(float(x) for x in centroid),len(indices),len(docs),radius,f"Cluster at depth {depth} containing {len(indices)} vectors",f"cluster-{depth}-{node_id[:6]}")
            nodes.append(node)
            if is_leaf: memberships[node_id]=[vector_ids[i] for i in indices]
            return node_id
        create(np.arange(len(vector_ids)),None,0,True)
        return BuiltTree(nodes,memberships)
    def _cluster(self, vectors: np.ndarray, requested: int) -> list[np.ndarray]:
        k=min(requested,len(vectors)); order=np.lexsort(vectors.T[::-1]); seeds=order[np.linspace(0,len(order)-1,k,dtype=int)]; centers=vectors[seeds].copy()
        for _ in range(20):
            assignment=np.argmax(vectors@centers.T,axis=1); new=[]
            for i in range(k):
                members=vectors[assignment==i]
                if not len(members): continue
                c=members.mean(0); new.append(c/max(np.linalg.norm(c),1e-12))
            if len(new)<2:return [np.arange(len(vectors))]
            updated=np.vstack(new)
            if updated.shape==centers.shape and np.allclose(updated,centers,atol=1e-5):break
            centers=updated; k=len(centers)
        return [np.flatnonzero(assignment==i) for i in range(k) if np.any(assignment==i)]
