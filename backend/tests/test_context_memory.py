from app.core.container import Container
from app.services.context_memory import ActiveContextService
from app.services.personal_knowledge import PersonalKnowledgeService

class FakeGeneration:
    def generate(self, prompt):
        if "Return JSON only: {\"sufficient\"" in prompt: return '{"sufficient": true}'
        if "private personal knowledge tree" in prompt: return '[{"path":["Projects","TreeMindDB"],"kind":"project","topic":"TreeMindDB","content":"TreeMindDB is a personal memory project.","fact_key":"project:treeminddb"}]'
        return "Active profile summary."

class StubLabeler:
    def label(self, notes, depth): return (notes[0]["topic"], "Knowledge summary.")

class NoLaya: available=False

def test_active_context_is_searchable_across_conversations_and_rolls_over(tmp_path):
    container=Container(str(tmp_path)); fake=FakeGeneration()
    context=ActiveContextService(container.context_collection.id,container.repository,container.vectors,container.embeddings,fake,window_tokens=20,retain_tokens=5)
    knowledge=PersonalKnowledgeService(container.personal_collection.id,container.repository,container.vectors,container.embeddings,NoLaya(),fake,StubLabeler())
    first=context.get_or_create(None); context.append(first.id,"user","TreeMindDB is a personal memory project with semantic search.")
    second=context.get_or_create(None); hits,sufficient=context.search(second.id,"What is TreeMindDB?")
    assert hits and hits[0].conversation_id == first.id and sufficient
    context.append(first.id,"assistant","I will retain that project description for later context.")
    assert context.needs_rollover(first.id)
    result=context.rollover(first.id,knowledge.curator,knowledge)
    assert result["archived_tokens"] > 0 and result["retained_tokens"] <= 5
    assert container.repository.current_personal_item(container.personal_collection.id,"project:treeminddb")
