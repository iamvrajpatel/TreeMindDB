from fastapi.testclient import TestClient
from app.main import create_app
from app.rag.generation import UnavailableGenerationProvider
from app.services.chat import ChatService
from app.services.chat_memory import ChatMemoryService
from app.services.memory import ClusterLabeler, MemoryAgent

class FakeModel:
    def generate(self, prompt):
        if "Extract only durable memory" in prompt: return '[{"kind":"fact","topic":"Testing","content":"Uses a synchronized memory tree."}]'
        if "Create a JSON object" in prompt: return '{"label":"Testing","summary":"Information about synchronized memory testing."}'
        return "The synchronized memory tree is ready."

def test_collection_ingest_search_and_trace(tmp_path):
    client=TestClient(create_app(str(tmp_path)))
    created=client.post("/api/v1/collections",json={"name":"demo","max_leaf_size":2,"minimum_split_size":2}).json()
    ingested=client.post(f"/api/v1/collections/{created['id']}/documents/text",json={"name":"guide.md","text":"Tree indexes embeddings for semantic retrieval.\n\nGraphs help search."})
    assert ingested.status_code == 201
    searched=client.post(f"/api/v1/collections/{created['id']}/search",json={"query":"semantic search","mode":"exact"})
    assert searched.status_code == 200
    trace=client.get(f"/api/v1/collections/{created['id']}/search/traces/{searched.json()['trace_id']}")
    assert trace.status_code == 200 and trace.json()["hits"]

def test_chat_is_explicit_when_generation_is_unconfigured(tmp_path):
    app=create_app(str(tmp_path)); app.state.container.generation=UnavailableGenerationProvider(); app.state.container.chat=ChatService(app.state.container.retrieval,app.state.container.generation)
    client=TestClient(app)
    collection=client.post("/api/v1/collections",json={"name":"chat"}).json()
    client.post(f"/api/v1/collections/{collection['id']}/documents/text",json={"text":"citation evidence"})
    response=client.post(f"/api/v1/collections/{collection['id']}/chat",json={"query":"evidence"})
    assert response.status_code == 200 and response.json()["generation_status"] == "unavailable"

def test_delete_document_rebuilds_tree(tmp_path):
    client=TestClient(create_app(str(tmp_path)))
    collection=client.post("/api/v1/collections",json={"name":"delete"}).json()
    document=client.post(f"/api/v1/collections/{collection['id']}/documents/text",json={"text":"temporary index entry"}).json()["document_id"]
    assert client.delete(f"/api/v1/collections/{collection['id']}/documents/{document}").status_code == 200
    assert client.get(f"/api/v1/collections/{collection['id']}/tree/stats").json()["vectors"] == 0

def test_chat_memory_status_and_tree_are_available(tmp_path):
    client=TestClient(create_app(str(tmp_path)))
    status=client.get("/api/v1/chat-memory/status")
    assert status.status_code == 200 and status.json()["collection_id"]
    assert client.get("/api/v1/chat-memory/tree").status_code == 200
    config=client.app.state.container.chat_memory_collection
    assert config.max_leaf_size == 4 and config.branching_factor == 3

def test_chat_stream_reports_memory_job_and_tree_completion(tmp_path):
    app=create_app(str(tmp_path)); container=app.state.container; model=FakeModel()
    container.generation=model; container.chat=ChatService(container.retrieval,model); container.memory_agent=MemoryAgent(model); container.labeler=ClusterLabeler(model)
    container.chat_memory=ChatMemoryService(container.chat_memory_collection.id,container.retrieval,container.chat,container.ingestion,container.memory_agent,container.labeler,container.repository)
    client=TestClient(app)
    with client.stream("POST","/api/v1/chat-memory/stream",json={"message":"Please remember the test design."}) as response:
        payload="".join(response.iter_text())
    assert response.status_code == 200
    for expected in ("answer_ready","memory_indexing","tree_rebuilding","memory_complete"):
        assert f"event: {expected}" in payload
    job_id=payload.split('"job_id": "')[1].split('"')[0]
    job=client.get(f"/api/v1/chat-memory/jobs/{job_id}").json()
    assert job["status"] == "complete" and job["note_count"] == 1
