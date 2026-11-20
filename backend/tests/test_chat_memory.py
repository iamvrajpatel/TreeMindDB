from app.core.container import Container
from app.services.chat import ChatService
from app.services.chat_memory import ChatMemoryService
from app.services.memory import ClusterLabeler, MemoryAgent

class FakeLocalModel:
    def generate(self, prompt: str) -> str:
        if "Extract only durable memory" in prompt:
            return '[{"kind":"preference","topic":"Database design","content":"Prefers a tree-based vector database."}]'
        if "Create a JSON object" in prompt:
            return '{"label":"Database design","summary":"Preferences and decisions about the database architecture."}'
        return "I will remember the database design preference. [source]"

def test_chat_stores_only_compact_memory_and_labels_tree(tmp_path):
    container=Container(str(tmp_path)); provider=FakeLocalModel()
    service=ChatMemoryService(container.chat_memory_collection.id,container.retrieval,ChatService(container.retrieval,provider),container.ingestion,MemoryAgent(provider),ClusterLabeler(provider),container.repository)
    response=service.send("I am thinking about a tree based database and might change my mind later.")
    assert response["memory_status"] == "saved"
    assert response["saved_notes"][0]["content"] == "Prefers a tree-based vector database."
    stored=container.repository.active_vectors(container.chat_memory_collection.id)
    assert len(stored) == 1
    assert "might change my mind" not in stored[0]["text"]
    tree=container.repository.get_nodes(container.chat_memory_collection.id)
    assert tree[0].label == "Database design"
