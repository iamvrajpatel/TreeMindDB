from app.core.container import Container
from app.services.personal_knowledge import PersonalKnowledgeService, TOP_LEVEL

class StubLabeler:
    def label(self, notes, depth): return (notes[0]["topic"], f"Knowledge about {notes[0]['topic']}.")

class UnavailableLaya:
    available=False
    reason="test"

def test_personal_tree_seeds_stable_owner_taxonomy(tmp_path):
    container=Container(str(tmp_path))
    service=PersonalKnowledgeService(container.personal_collection.id,container.repository,container.vectors,container.embeddings,UnavailableLaya(),container.generation,StubLabeler())
    nodes=service.tree()
    root=next(node for node in nodes if node.parent_id is None)
    assert root.label == "My Personal Knowledge"
    assert {node.label for node in nodes if node.parent_id==root.id} == set(TOP_LEVEL)

def test_personal_knowledge_creates_branch_retrieves_and_requires_conflict_confirmation(tmp_path):
    container=Container(str(tmp_path)); service=PersonalKnowledgeService(container.personal_collection.id,container.repository,container.vectors,container.embeddings,UnavailableLaya(),container.generation,StubLabeler())
    created,added,pending=service.persist_items([{"path":["Projects","TreeMindDB"],"kind":"project","topic":"TreeMindDB","content":"TreeMindDB is a personal knowledge system.","fact_key":"project:treeminddb:purpose"}])
    assert [node.label for node in created] == ["TreeMindDB"] and len(added)==1 and not pending
    retrieval=service.retrieve("What is TreeMindDB?")
    assert retrieval.fallback and retrieval.hits[0].topic == "TreeMindDB"
    _,_,pending=service.persist_items([{"path":["Projects","TreeMindDB"],"kind":"project","topic":"TreeMindDB","content":"TreeMindDB is a project tracker.","fact_key":"project:treeminddb:purpose"}])
    assert len(pending)==1
    update=service.confirm_update(pending[0],True)
    assert update and container.repository.current_personal_item(container.personal_collection.id,"project:treeminddb:purpose")["content"] == "TreeMindDB is a project tracker."

class ResumeGeneration:
    def generate(self, prompt):
        return '[{"path":["Experience & Skills","Backend"],"kind":"experience","topic":"Python","content":"The owner has professional Python backend experience.","fact_key":"experience:python-backend"},{"path":["Projects","TreeMindDB"],"kind":"project","topic":"TreeMindDB","content":"The owner is building TreeMindDB.","fact_key":"project:treeminddb"}]'

def test_resume_source_is_retained_and_split_into_atomic_facts(tmp_path):
    container=Container(str(tmp_path)); service=PersonalKnowledgeService(container.personal_collection.id,container.repository,container.vectors,container.embeddings,UnavailableLaya(),ResumeGeneration(),StubLabeler())
    result=service.ingest_source("Resume: Python backend engineer. Built TreeMindDB.","resume.txt")
    assert result["source_document_id"] and len(result["added_item_ids"]) == 2
    projects=next(node for node in service.tree() if node.label == "Projects")
    details=container.repository.personal_node_details(container.personal_collection.id,projects.id)
    child=next(node for node in details["children"] if node.label == "TreeMindDB")
    fact=container.repository.personal_node_details(container.personal_collection.id,child.id)["current_facts"][0]
    assert fact["source_document_id"] == result["source_document_id"] and "Resume:" in fact["source_excerpt"]
