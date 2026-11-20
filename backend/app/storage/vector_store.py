from __future__ import annotations
from pathlib import Path
from collections.abc import Sequence
import numpy as np

class BinaryVectorStore:
    def __init__(self, root: str): self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
    def _path(self, collection_id: str) -> Path: return self.root / f"{collection_id}.vectors"
    def append(self, collection_id: str, vectors: np.ndarray) -> list[int]:
        path = self._path(collection_id)
        offsets = list(range(path.stat().st_size // 4 if path.exists() else 0, (path.stat().st_size // 4 if path.exists() else 0) + vectors.size, vectors.shape[1]))
        staging = path.with_suffix(".staging")
        with staging.open("wb") as f:
            vectors.astype(np.float32).tofile(f); f.flush()
        with path.open("ab") as target, staging.open("rb") as source:
            target.write(source.read()); target.flush()
        staging.unlink(missing_ok=True)
        return offsets
    def read(self, collection_id: str, offsets: Sequence[int], dimension: int) -> np.ndarray:
        if not offsets: return np.empty((0, dimension), dtype=np.float32)
        data = np.memmap(self._path(collection_id), dtype=np.float32, mode="r")
        return np.vstack([np.array(data[o:o + dimension]) for o in offsets])
