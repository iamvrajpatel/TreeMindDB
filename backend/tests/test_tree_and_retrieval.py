from app.core.container import Container
from app.domain.models import SearchMode

def setup_collection(tmp_path, max_leaf_size=2):
    c=Container(str(tmp_path)); collection=c.repository.create_collection("test",max_leaf_size=max_leaf_size,minimum_split_size=2)
    return c,collection

def test_builds_persistent_tree_and_exact_search(tmp_path):
    c,collection=setup_collection(tmp_path)
    for name,text in [("vectors.txt",b"alpha embeddings vector database"),("fruit.txt",b"banana orchard fruit harvest"),("search.txt",b"vector retrieval semantic search")]:
        c.ingestion.ingest_bytes(collection.id,name,text)
    assert len(c.repository.get_nodes(collection.id)) >= 2
    result=c.retrieval.search(collection.id,"semantic vector search",SearchMode.EXACT)
    assert result.hits and result.candidate_vectors == 3
    restarted=Container(str(tmp_path))
    assert restarted.retrieval.search(collection.id,"semantic vector search",SearchMode.EXACT).hits[0].chunk_id == result.hits[0].chunk_id

def test_laya_unavailable_falls_back_to_centroid(tmp_path):
    c,collection=setup_collection(tmp_path)
    for name,text in [("vectors.txt",b"one vector database"),("fruit.txt",b"two fruit orchard"),("ml.txt",b"three machine learning")]:
        c.ingestion.ingest_bytes(collection.id,name,text)
    c.laya.available=False; c.laya.reason="test unavailable"
    result=c.retrieval.search(collection.id,"vector",SearchMode.LAYA,beam_width=2)
    assert result.fallback
    assert result.steps and result.steps[0].strategy == "exact-fallback"

def test_invalid_collection_model_dimension_is_rejected(tmp_path):
    c,collection=setup_collection(tmp_path)
    c.repository.conn.execute("UPDATE collections SET config=? WHERE id=?", ('{"id":"'+collection.id+'","name":"test","embedding_dimension":12}',collection.id)); c.repository.conn.commit()
    try: c.ingestion.ingest_bytes(collection.id,"x.txt",b"incompatible")
    except ValueError as exc: assert "dimension" in str(exc)
    else: raise AssertionError("expected incompatible model dimensions to be rejected")
