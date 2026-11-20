from __future__ import annotations
import asyncio,json,os
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from app.core.container import Container
from app.domain.models import SearchMode
from app.rag.generation import UnavailableGenerationProvider
from app.schemas.api import CollectionCreate,TextIngest,SearchRequest,ChatRequest,ChatMemoryMessage

router=APIRouter(prefix="/api/v1")
def container(request:Request)->Container:return request.app.state.container
def missing(exc:Exception):
    if isinstance(exc,KeyError): raise HTTPException(404,str(exc))
    if isinstance(exc,ValueError): raise HTTPException(422,str(exc))
    raise exc

@router.get("/health")
def health(c:Container=Depends(container)): return {"status":"ok","embedding_model":c.embeddings.model_id,"embedding":c.embedding_status,"laya_available":c.laya.available}
@router.get("/models/status")
def models(c:Container=Depends(container)): return {"embedding":{"id":c.embeddings.model_id,"dimension":c.embeddings.dimension,**c.embedding_status},"laya":{"available":c.laya.available,"state":c.laya.state,"reason":c.laya.reason,"device":c.settings.laya_device or "auto"},"generation":{"configured":not isinstance(c.generation,UnavailableGenerationProvider),"provider":"ollama" if c.settings.ollama_model else None,"model":c.settings.ollama_model,"base_url":c.settings.ollama_base_url}}
@router.get("/config")
def config(): return {"cors_origins":os.getenv("CORS_ORIGINS","http://localhost:3000"),"max_upload_bytes":25*1024*1024}

@router.get("/chat-memory/status")
def chat_memory_status(c:Container=Depends(container)):
    return {"collection_id":c.chat_memory_collection.id,"laya_available":c.laya.available,"memory_agent_available":c.memory_agent.available,"generation_configured":not isinstance(c.generation,UnavailableGenerationProvider)}
@router.get("/personal-tree")
def personal_tree(c:Container=Depends(container)): return c.personal_knowledge.tree()
@router.post("/personal-tree/sources/text",status_code=201)
def ingest_personal_source(payload:TextIngest,c:Container=Depends(container)):
    """Explicit source ingestion for pasted resumes and long-lived documents."""
    return c.personal_knowledge.ingest_source(payload.text,payload.name)
@router.post("/personal-tree/sources/upload",status_code=201)
async def upload_personal_source(file:UploadFile=File(...),c:Container=Depends(container)):
    content=await file.read()
    if len(content)>25*1024*1024: raise HTTPException(413,"Upload exceeds 25 MB limit")
    name=file.filename or "upload.txt"
    pages=c.ingestion.extractor.extract(content,name)
    text="\n\n".join(page for page,_ in pages)
    if not text.strip(): raise HTTPException(422,"No extractable text found")
    return c.personal_knowledge.ingest_source(text,name)
@router.get("/conversations")
def conversations(c:Container=Depends(container)): return c.repository.conversations()
@router.post("/conversations",status_code=201)
def create_conversation(c:Container=Depends(container)): return c.repository.create_conversation()
@router.get("/conversations/{conversation_id}/sync-status")
def sync_status(conversation_id:str,c:Container=Depends(container)): return c.repository.conversation_sync_status(conversation_id)
@router.post("/conversations/{conversation_id}/sync/stream")
def sync_conversation(conversation_id:str,c:Container=Depends(container)):
    def events():
        for item in c.context_assistant.sync(conversation_id): yield f"event: {item['event']}\ndata: {json.dumps(item)}\n\n"
    return StreamingResponse(events(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
@router.get("/conversations/{conversation_id}/context")
def conversation_context(conversation_id:str,c:Container=Depends(container)):
    conversation=c.repository.conversation(conversation_id)
    if not conversation: raise HTTPException(404,"Conversation not found")
    return {"conversation":conversation,"window_tokens":c.settings.context_window_tokens,"retain_tokens":c.settings.context_retain_tokens,"recent_segments":[{"role":item["role"],"text":item["text"],"token_count":item["token_count"]} for item in c.repository.recent_context_records(conversation_id,10)]}
@router.post("/context-assistant/stream")
def context_assistant_stream(payload:ChatMemoryMessage,c:Container=Depends(container)):
    def events():
        for item in c.context_assistant.stream(payload.message,payload.conversation_id,payload.top_k): yield f"event: {item['event']}\ndata: {json.dumps(item)}\n\n"
    return StreamingResponse(events(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
@router.get("/personal-tree/nodes/{node_id}")
def personal_tree_node(node_id:str,c:Container=Depends(container)):
    details=c.repository.personal_node_details(c.personal_collection.id,node_id)
    if not details: raise HTTPException(404,"Personal knowledge node not found")
    return details
@router.get("/traces")
def traces(limit:int=50,c:Container=Depends(container)): return c.repository.recent_operation_traces(min(max(limit,1),200))
@router.get("/traces/{trace_id}")
def trace_detail(trace_id:str,c:Container=Depends(container)):
    trace=c.repository.operation_trace(trace_id)
    if not trace: raise HTTPException(404,"Trace not found")
    return trace
@router.post("/personal-assistant/stream")
def personal_assistant_stream(payload:ChatMemoryMessage,c:Container=Depends(container)):
    def events():
        for item in c.personal_assistant.stream(payload.message,payload.top_k): yield f"event: {item['event']}\ndata: {json.dumps(item)}\n\n"
    return StreamingResponse(events(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
@router.post("/personal-tree/updates/{update_id}/confirm")
def confirm_personal_update(update_id:str,c:Container=Depends(container)):
    result=c.personal_knowledge.confirm_update(update_id,True)
    if not result: raise HTTPException(404,"Pending update not found")
    return {"status":"approved","update":result}
@router.post("/personal-tree/updates/{update_id}/reject")
def reject_personal_update(update_id:str,c:Container=Depends(container)):
    result=c.personal_knowledge.confirm_update(update_id,False)
    if not result: raise HTTPException(404,"Pending update not found")
    return {"status":"rejected","update":result}
@router.get("/chat-memory/tree")
def chat_memory_tree(c:Container=Depends(container)):
    return c.repository.get_nodes(c.chat_memory_collection.id)
@router.get("/chat-memory/nodes/{node_id}")
def chat_memory_node(node_id:str,c:Container=Depends(container)):
    node=next((item for item in c.repository.get_nodes(c.chat_memory_collection.id) if item.id==node_id),None)
    if not node: raise HTTPException(404,"Node not found")
    return {"node":node,"notes":c.repository.notes_for_node(node_id)}
@router.get("/chat-memory/traces/{trace_id}")
def chat_memory_trace(trace_id:str,c:Container=Depends(container)):
    result=c.repository.get_trace(trace_id)
    if not result:raise HTTPException(404,"Trace not found")
    return result
@router.post("/chat-memory/message")
def chat_memory_message(payload:ChatMemoryMessage,c:Container=Depends(container)):
    try:return c.chat_memory.send(payload.message,payload.top_k)
    except Exception as exc: missing(exc)
@router.post("/chat-memory/stream")
def chat_memory_stream(payload:ChatMemoryMessage,c:Container=Depends(container)):
    def events():
        for item in c.chat_memory.stream(payload.message,payload.top_k):
            yield f"event: {item['event']}\ndata: {json.dumps(item)}\n\n"
    return StreamingResponse(events(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
@router.get("/chat-memory/jobs/{job_id}")
def chat_memory_job(job_id:str,c:Container=Depends(container)):
    job=c.repository.get_memory_job(job_id)
    if not job: raise HTTPException(404,"Memory job not found")
    return job

@router.post("/collections",status_code=201)
def create_collection(payload:CollectionCreate,c:Container=Depends(container)):
    try:
        config=payload.model_dump(exclude_none=True)
        config.setdefault("embedding_dimension",c.embeddings.dimension)
        config["embedding_model"]=c.embeddings.model_id
        return c.repository.create_collection(**config)
    except Exception as exc: missing(exc)
@router.get("/collections")
def list_collections(c:Container=Depends(container)): return c.repository.list_collections()

@router.post("/collections/{collection_id}/documents/text",status_code=201)
def ingest_text(collection_id:str,payload:TextIngest,c:Container=Depends(container)):
    try:return c.ingestion.ingest_bytes(collection_id,payload.name,payload.text.encode())
    except Exception as exc: missing(exc)
@router.post("/collections/{collection_id}/documents/upload",status_code=201)
async def upload(collection_id:str,file:UploadFile=File(...),c:Container=Depends(container)):
    try:return c.ingestion.ingest_bytes(collection_id,file.filename or "upload.txt",await file.read())
    except Exception as exc: missing(exc)
@router.get("/collections/{collection_id}/documents")
def documents(collection_id:str,c:Container=Depends(container)):
    if not c.repository.get_collection(collection_id):raise HTTPException(404,"Collection not found")
    return [{"id":r[0],"name":r[1],"created_at":r[2]} for r in c.repository.conn.execute("SELECT id,name,created_at FROM documents WHERE collection_id=? ORDER BY created_at DESC",(collection_id,))]
@router.delete("/collections/{collection_id}/documents/{document_id}")
def delete_document(collection_id:str,document_id:str,c:Container=Depends(container)):
    try:return c.ingestion.delete_document(collection_id,document_id)
    except Exception as exc: missing(exc)

@router.post("/collections/{collection_id}/tree/rebuild")
def rebuild(collection_id:str,c:Container=Depends(container)):
    try:return c.ingestion.rebuild(collection_id)
    except Exception as exc: missing(exc)
@router.get("/collections/{collection_id}/tree")
def tree(collection_id:str,c:Container=Depends(container)): return c.repository.get_nodes(collection_id)
@router.get("/collections/{collection_id}/tree/stats")
def stats(collection_id:str,c:Container=Depends(container)):
    if not c.repository.get_collection(collection_id):raise HTTPException(404,"Collection not found")
    return c.repository.stats(collection_id)
@router.get("/collections/{collection_id}/tree/nodes/{node_id}")
def node(collection_id:str,node_id:str,c:Container=Depends(container)):
    item=next((n for n in c.repository.get_nodes(collection_id) if n.id==node_id),None)
    if not item:raise HTTPException(404,"Node not found")
    return item

@router.post("/collections/{collection_id}/search")
def search(collection_id:str,payload:SearchRequest,c:Container=Depends(container)):
    try:return c.retrieval.search(collection_id,**payload.model_dump())
    except Exception as exc: missing(exc)
@router.post("/collections/{collection_id}/search/compare")
def compare(collection_id:str,payload:SearchRequest,c:Container=Depends(container)):
    try:return {mode.value:c.retrieval.search(collection_id,payload.query,mode,payload.top_k,payload.beam_width) for mode in SearchMode}
    except Exception as exc: missing(exc)
@router.get("/collections/{collection_id}/search/traces/{trace_id}")
def trace(collection_id:str,trace_id:str,c:Container=Depends(container)):
    result=c.repository.get_trace(trace_id)
    if not result:raise HTTPException(404,"Trace not found")
    return result

@router.post("/collections/{collection_id}/chat")
def chat(collection_id:str,payload:ChatRequest,c:Container=Depends(container)):
    try:
        return c.chat.answer(collection_id,payload.query,payload.mode,payload.top_k,payload.beam_width)
    except Exception as exc: missing(exc)
@router.post("/collections/{collection_id}/chat/stream")
async def chat_stream(collection_id:str,payload:ChatRequest,c:Container=Depends(container)):
    async def events():
        yield "event: retrieval_started\ndata: {}\n\n"
        response=c.chat.answer(collection_id,payload.query,payload.mode,payload.top_k,payload.beam_width); result=response["retrieval"]
        yield "event: retrieval_complete\ndata: "+json.dumps({"trace_id":result.trace_id,"hits":[h.__dict__ for h in result.hits]})+"\n\n"
        yield f"event: {response['generation_status']}\ndata: "+json.dumps({"answer":response["answer"],"citations":response["citations"]})+"\n\n"
    return StreamingResponse(events(),media_type="text/event-stream")
