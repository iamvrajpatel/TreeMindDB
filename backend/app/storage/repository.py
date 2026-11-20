from __future__ import annotations
import json, sqlite3, uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable
from app.domain.models import CollectionConfig, TreeNode, NodeType

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS collections (id TEXT PRIMARY KEY,name TEXT UNIQUE NOT NULL,config TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY,collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,name TEXT NOT NULL,content_hash TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY,document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,text TEXT NOT NULL,position INTEGER NOT NULL,page INTEGER,token_count INTEGER NOT NULL,metadata TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY,collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,offset INTEGER NOT NULL,active INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY,collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,parent_id TEXT,node_type TEXT NOT NULL,depth INTEGER NOT NULL,child_ids TEXT NOT NULL,centroid TEXT NOT NULL,vector_count INTEGER NOT NULL,document_count INTEGER NOT NULL,radius REAL NOT NULL,summary TEXT NOT NULL,label TEXT);
CREATE TABLE IF NOT EXISTS memberships (node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,vector_id TEXT NOT NULL REFERENCES vectors(id) ON DELETE CASCADE,PRIMARY KEY(node_id,vector_id));
CREATE TABLE IF NOT EXISTS traces (id TEXT PRIMARY KEY,collection_id TEXT NOT NULL,payload TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS index_operations (id TEXT PRIMARY KEY,collection_id TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memory_notes (id TEXT PRIMARY KEY,collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,kind TEXT NOT NULL,topic TEXT NOT NULL,content TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memory_jobs (id TEXT PRIMARY KEY,collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,status TEXT NOT NULL,detail TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,changed_node_ids TEXT NOT NULL DEFAULT '[]',note_count INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS personal_nodes (id TEXT PRIMARY KEY,collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,parent_id TEXT REFERENCES personal_nodes(id) ON DELETE CASCADE,label TEXT NOT NULL,summary TEXT NOT NULL,depth INTEGER NOT NULL,node_type TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,UNIQUE(collection_id,parent_id,label));
CREATE TABLE IF NOT EXISTS personal_items (id TEXT PRIMARY KEY,collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,node_id TEXT NOT NULL REFERENCES personal_nodes(id) ON DELETE CASCADE,chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,kind TEXT NOT NULL,topic TEXT NOT NULL,content TEXT NOT NULL,fact_key TEXT NOT NULL,status TEXT NOT NULL,supersedes_id TEXT REFERENCES personal_items(id),source_document_id TEXT REFERENCES documents(id),source_excerpt TEXT NOT NULL DEFAULT '',created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_personal_items_key ON personal_items(collection_id,fact_key,status);
CREATE TABLE IF NOT EXISTS personal_updates (id TEXT PRIMARY KEY,collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,existing_item_id TEXT NOT NULL REFERENCES personal_items(id) ON DELETE CASCADE,proposed_node_id TEXT NOT NULL REFERENCES personal_nodes(id) ON DELETE CASCADE,kind TEXT NOT NULL,topic TEXT NOT NULL,content TEXT NOT NULL,fact_key TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL,resolved_at TEXT);
CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY,title TEXT NOT NULL,active_tokens INTEGER NOT NULL DEFAULT 0,profile_summary TEXT NOT NULL DEFAULT '',created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS context_segments (id TEXT PRIMARY KEY,conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,sequence INTEGER NOT NULL,role TEXT NOT NULL,text TEXT NOT NULL,token_count INTEGER NOT NULL,active INTEGER NOT NULL DEFAULT 1,created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_context_segments_active ON context_segments(active,conversation_id,sequence);
CREATE TABLE IF NOT EXISTS operation_traces (id TEXT PRIMARY KEY,conversation_id TEXT,operation_type TEXT NOT NULL,status TEXT NOT NULL,events TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS conversation_syncs (conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,last_sequence INTEGER NOT NULL DEFAULT 0,status TEXT NOT NULL DEFAULT 'idle',updated_at TEXT NOT NULL);
"""

class SqliteTreeRepository:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True); self.conn = sqlite3.connect(path, check_same_thread=False); self.conn.row_factory = sqlite3.Row; self.conn.executescript(SCHEMA); self._migrate(); self.conn.commit()
    def _migrate(self):
        """Small, idempotent migrations for databases created by earlier MVP builds."""
        columns={row["name"] for row in self.conn.execute("PRAGMA table_info(personal_items)")}
        if "source_document_id" not in columns:self.conn.execute("ALTER TABLE personal_items ADD COLUMN source_document_id TEXT")
        if "source_excerpt" not in columns:self.conn.execute("ALTER TABLE personal_items ADD COLUMN source_excerpt TEXT NOT NULL DEFAULT ''")
    @staticmethod
    def _now(): return datetime.now(timezone.utc).isoformat()
    def create_collection(self, name: str, **config) -> CollectionConfig:
        item = CollectionConfig(id=str(uuid.uuid4()), name=name, **config); self.conn.execute("INSERT INTO collections VALUES (?,?,?,?)", (item.id, name, json.dumps(item.__dict__), self._now())); self.conn.commit(); return item
    def list_collections(self): return [CollectionConfig(**json.loads(r["config"])) for r in self.conn.execute("SELECT config FROM collections ORDER BY name")]
    def get_or_create_collection(self, name: str, **config) -> CollectionConfig:
        row=self.conn.execute("SELECT config FROM collections WHERE name=?",(name,)).fetchone()
        return CollectionConfig(**json.loads(row[0])) if row else self.create_collection(name,**config)
    def update_collection_config(self, collection: CollectionConfig) -> CollectionConfig:
        with self.conn:self.conn.execute("UPDATE collections SET config=? WHERE id=?",(json.dumps(collection.__dict__),collection.id))
        return collection
    def get_collection(self, collection_id: str):
        row = self.conn.execute("SELECT config FROM collections WHERE id=?", (collection_id,)).fetchone(); return CollectionConfig(**json.loads(row[0])) if row else None
    def insert_document(self, collection_id: str, name: str, content_hash: str, chunks: list[tuple[str,int|None,int]], offsets: list[int]) -> tuple[str, list[str]]:
        doc_id = str(uuid.uuid4()); ids=[]
        with self.conn:
            self.conn.execute("INSERT INTO documents VALUES (?,?,?,?,?)", (doc_id,collection_id,name,content_hash,self._now()))
            for pos, ((text,page,tokens), offset) in enumerate(zip(chunks, offsets)):
                chunk_id, vector_id = str(uuid.uuid4()), str(uuid.uuid4()); ids.append(chunk_id)
                self.conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)", (chunk_id,doc_id,text,pos,page,tokens,"{}"))
                self.conn.execute("INSERT INTO vectors VALUES (?,?,?,?,1)", (vector_id,collection_id,chunk_id,offset))
        return doc_id, ids
    def active_vectors(self, collection_id: str):
        return self.conn.execute("SELECT v.id vector_id,v.chunk_id,v.offset,c.document_id,c.text,d.name FROM vectors v JOIN chunks c ON c.id=v.chunk_id JOIN documents d ON d.id=c.document_id WHERE v.collection_id=? AND v.active=1 ORDER BY v.id",(collection_id,)).fetchall()
    def replace_tree(self, collection_id: str, nodes: Iterable[TreeNode], memberships: dict[str,list[str]]):
        with self.conn:
            self.conn.execute("DELETE FROM nodes WHERE collection_id=?",(collection_id,))
            for n in nodes:
                self.conn.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",(n.id,n.collection_id,n.parent_id,n.node_type,n.depth,json.dumps(n.child_ids),json.dumps(n.centroid),n.vector_count,n.document_count,n.radius,n.summary,n.label))
                self.conn.executemany("INSERT INTO memberships VALUES (?,?)",((n.id, v) for v in memberships.get(n.id,[])))
    def get_nodes(self, collection_id: str):
        rows=self.conn.execute("SELECT * FROM nodes WHERE collection_id=? ORDER BY depth,id",(collection_id,)); return [TreeNode(id=r['id'],collection_id=r['collection_id'],parent_id=r['parent_id'],node_type=NodeType(r['node_type']),depth=r['depth'],child_ids=tuple(json.loads(r['child_ids'])),centroid=tuple(json.loads(r['centroid'])),vector_count=r['vector_count'],document_count=r['document_count'],radius=r['radius'],summary=r['summary'],label=r['label']) for r in rows]
    def leaf_vector_ids(self, node_ids: list[str]):
        if not node_ids:return []
        placeholders=','.join('?'*len(node_ids)); return [r[0] for r in self.conn.execute(f"SELECT DISTINCT vector_id FROM memberships WHERE node_id IN ({placeholders})",node_ids)]
    def vector_records(self, vector_ids: list[str]):
        if not vector_ids:return []
        p=','.join('?'*len(vector_ids)); return self.conn.execute(f"SELECT v.id vector_id,v.chunk_id,v.offset,c.document_id,c.text,d.name FROM vectors v JOIN chunks c ON c.id=v.chunk_id JOIN documents d ON d.id=c.document_id WHERE v.id IN ({p})",vector_ids).fetchall()
    def save_trace(self, collection_id: str, trace_id: str, payload: dict):
        self.conn.execute("INSERT INTO traces VALUES (?,?,?,?)",(trace_id,collection_id,json.dumps(payload),self._now())); self.conn.commit()
    def get_trace(self, trace_id: str):
        row=self.conn.execute("SELECT payload FROM traces WHERE id=?",(trace_id,)).fetchone(); return json.loads(row[0]) if row else None
    def stats(self, collection_id: str):
        q=lambda sql:self.conn.execute(sql,(collection_id,)).fetchone()[0]
        return {"documents":q("SELECT count(*) FROM documents WHERE collection_id=?"),"vectors":q("SELECT count(*) FROM vectors WHERE collection_id=? AND active=1"),"nodes":q("SELECT count(*) FROM nodes WHERE collection_id=?"),"leaves":q("SELECT count(*) FROM nodes WHERE collection_id=? AND node_type='leaf'")}
    def delete_document(self, collection_id: str, document_id: str) -> bool:
        with self.conn:
            row=self.conn.execute("SELECT id FROM documents WHERE id=? AND collection_id=?",(document_id,collection_id)).fetchone()
            if not row:return False
            # Preserve vector bytes for crash-safe append storage; remove references atomically.
            self.conn.execute("DELETE FROM documents WHERE id=?",(document_id,))
        return True
    def save_memory_notes(self, collection_id: str, notes: list[dict], chunk_ids: list[str]) -> list[dict]:
        saved=[]
        with self.conn:
            for note,chunk_id in zip(notes,chunk_ids):
                item={"id":str(uuid.uuid4()),"collection_id":collection_id,"chunk_id":chunk_id,"kind":note["kind"],"topic":note["topic"],"content":note["content"],"created_at":self._now()}
                self.conn.execute("INSERT INTO memory_notes VALUES (?,?,?,?,?,?,?)",tuple(item.values())); saved.append(item)
        return saved
    def notes_for_vector_ids(self, vector_ids: list[str]) -> list[dict]:
        if not vector_ids:return []
        p=','.join('?'*len(vector_ids)); rows=self.conn.execute(f"SELECT n.id,n.kind,n.topic,n.content,n.chunk_id FROM memory_notes n JOIN vectors v ON v.chunk_id=n.chunk_id WHERE v.id IN ({p})",vector_ids)
        return [dict(row) for row in rows]
    def notes_for_node(self, node_id: str) -> list[dict]:
        rows=self.conn.execute("SELECT n.id,n.kind,n.topic,n.content,n.chunk_id FROM memory_notes n JOIN vectors v ON v.chunk_id=n.chunk_id JOIN memberships m ON m.vector_id=v.id WHERE m.node_id=? ORDER BY n.created_at DESC LIMIT 20",(node_id,))
        return [dict(row) for row in rows]
    def create_memory_job(self, collection_id: str) -> str:
        job_id=str(uuid.uuid4()); now=self._now()
        with self.conn:self.conn.execute("INSERT INTO memory_jobs (id,collection_id,status,detail,created_at,updated_at) VALUES (?,?,?,?,?,?)",(job_id,collection_id,"queued","Waiting for reply",now,now))
        return job_id
    def update_memory_job(self, job_id: str, status: str, detail: str, changed_node_ids: list[str] | None = None, note_count: int | None = None):
        fields=["status=?","detail=?","updated_at=?"]; values=[status,detail,self._now()]
        if changed_node_ids is not None: fields.append("changed_node_ids=?"); values.append(json.dumps(changed_node_ids))
        if note_count is not None: fields.append("note_count=?"); values.append(note_count)
        values.append(job_id)
        with self.conn:self.conn.execute(f"UPDATE memory_jobs SET {','.join(fields)} WHERE id=?",values)
    def get_memory_job(self, job_id: str):
        row=self.conn.execute("SELECT * FROM memory_jobs WHERE id=?",(job_id,)).fetchone()
        if not row:return None
        item=dict(row); item["changed_node_ids"]=json.loads(item["changed_node_ids"]); return item
    def personal_nodes(self, collection_id: str):
        rows=self.conn.execute("SELECT * FROM personal_nodes WHERE collection_id=? ORDER BY depth,label",(collection_id,))
        from app.domain.personal import PersonalNode
        return [PersonalNode(r["id"],r["parent_id"],r["label"],r["summary"],r["depth"],r["node_type"],r["created_at"],r["updated_at"]) for r in rows]
    def create_personal_node(self, collection_id: str, parent_id: str | None, label: str, summary: str, depth: int, node_type: str):
        row=self.conn.execute("SELECT * FROM personal_nodes WHERE collection_id=? AND parent_id IS ? AND label=?",(collection_id,parent_id,label)).fetchone()
        if row:
            from app.domain.personal import PersonalNode
            return PersonalNode(row["id"],row["parent_id"],row["label"],row["summary"],row["depth"],row["node_type"],row["created_at"],row["updated_at"])
        from app.domain.personal import PersonalNode
        item=PersonalNode(str(uuid.uuid4()),parent_id,label[:80],summary[:400],depth,node_type,self._now(),self._now())
        with self.conn:self.conn.execute("INSERT INTO personal_nodes VALUES (?,?,?,?,?,?,?,?,?)",(item.id,collection_id,item.parent_id,item.label,item.summary,item.depth,item.node_type,item.created_at,item.updated_at))
        return item
    def reparent_personal_node(self, node_id: str, parent_id: str, depth: int):
        with self.conn:self.conn.execute("UPDATE personal_nodes SET parent_id=?,depth=?,updated_at=? WHERE id=?",(parent_id,depth,self._now(),node_id))
    def delete_personal_node(self, node_id: str):
        with self.conn:self.conn.execute("DELETE FROM personal_nodes WHERE id=?",(node_id,))
    def update_personal_node_summary(self, node_id: str, summary: str):
        with self.conn:self.conn.execute("UPDATE personal_nodes SET summary=?,updated_at=? WHERE id=?",(summary[:400],self._now(),node_id))
    def personal_children(self, collection_id: str, parent_id: str):
        return [node for node in self.personal_nodes(collection_id) if node.parent_id==parent_id]
    def personal_node(self, node_id: str):
        return next((node for node in self.personal_nodes_for_node(node_id)),None)
    def personal_nodes_for_node(self, node_id: str):
        row=self.conn.execute("SELECT * FROM personal_nodes WHERE id=?",(node_id,)).fetchone()
        if not row:return []
        from app.domain.personal import PersonalNode
        return [PersonalNode(row["id"],row["parent_id"],row["label"],row["summary"],row["depth"],row["node_type"],row["created_at"],row["updated_at"])]
    def insert_personal_item(self, collection_id: str, node_id: str, chunk_id: str, item: dict, status: str = "current", supersedes_id: str | None = None, source_document_id: str | None = None, source_excerpt: str = ""):
        item_id=str(uuid.uuid4()); now=self._now()
        with self.conn:self.conn.execute("INSERT INTO personal_items (id,collection_id,node_id,chunk_id,kind,topic,content,fact_key,status,supersedes_id,source_document_id,source_excerpt,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(item_id,collection_id,node_id,chunk_id,item["kind"],item["topic"],item["content"],item["fact_key"],status,supersedes_id,source_document_id,source_excerpt[:1200],now,now))
        return item_id
    def current_personal_item(self, collection_id: str, fact_key: str):
        row=self.conn.execute("SELECT * FROM personal_items WHERE collection_id=? AND fact_key=? AND status='current' ORDER BY updated_at DESC LIMIT 1",(collection_id,fact_key)).fetchone()
        return dict(row) if row else None
    def personal_items_for_nodes(self, collection_id: str, node_ids: list[str]):
        if not node_ids:return []
        p=','.join('?'*len(node_ids)); rows=self.conn.execute(f"SELECT p.*,v.id vector_id,v.offset FROM personal_items p JOIN vectors v ON v.chunk_id=p.chunk_id WHERE p.collection_id=? AND p.status='current' AND p.node_id IN ({p})",[collection_id,*node_ids])
        return [dict(row) for row in rows]
    def all_personal_items(self, collection_id: str):
        rows=self.conn.execute("SELECT p.*,v.id vector_id,v.offset FROM personal_items p JOIN vectors v ON v.chunk_id=p.chunk_id WHERE p.collection_id=? AND p.status='current'",(collection_id,))
        return [dict(row) for row in rows]
    def create_personal_update(self, collection_id: str, existing_item_id: str, node_id: str, item: dict):
        update_id=str(uuid.uuid4()); now=self._now()
        with self.conn:self.conn.execute("INSERT INTO personal_updates VALUES (?,?,?,?,?,?,?,?,?,?,?)",(update_id,collection_id,existing_item_id,node_id,item["kind"],item["topic"],item["content"],item["fact_key"],"pending",now,None))
        return update_id
    def personal_update(self, update_id: str):
        row=self.conn.execute("SELECT * FROM personal_updates WHERE id=?",(update_id,)).fetchone(); return dict(row) if row else None
    def resolve_personal_update(self, update_id: str, approved: bool):
        update=self.personal_update(update_id)
        if not update or update["status"]!="pending": return None
        now=self._now()
        with self.conn:
            self.conn.execute("UPDATE personal_updates SET status=?,resolved_at=? WHERE id=?",("approved" if approved else "rejected",now,update_id))
            if approved:self.conn.execute("UPDATE personal_items SET status='superseded',updated_at=? WHERE id=?",(now,update["existing_item_id"]))
        return update
    def create_conversation(self, title: str = "New conversation"):
        from app.domain.context import Conversation
        item=Conversation(str(uuid.uuid4()),title[:120],0,"",self._now(),self._now())
        with self.conn:self.conn.execute("INSERT INTO conversations VALUES (?,?,?,?,?,?)",(item.id,item.title,item.active_tokens,item.profile_summary,item.created_at,item.updated_at))
        return item
    def conversations(self):
        from app.domain.context import Conversation
        rows=self.conn.execute("SELECT * FROM conversations ORDER BY updated_at DESC")
        return [Conversation(row["id"],row["title"],row["active_tokens"],row["profile_summary"],row["created_at"],row["updated_at"]) for row in rows]
    def conversation(self, conversation_id: str):
        return next((item for item in self.conversations() if item.id==conversation_id),None)
    def update_conversation(self, conversation_id: str, active_tokens: int | None = None, profile_summary: str | None = None, title: str | None = None):
        fields=["updated_at=?"]; values=[self._now()]
        if active_tokens is not None:fields.append("active_tokens=?");values.append(active_tokens)
        if profile_summary is not None:fields.append("profile_summary=?");values.append(profile_summary[:1000])
        if title is not None:fields.append("title=?");values.append(title[:120])
        values.append(conversation_id)
        with self.conn:self.conn.execute(f"UPDATE conversations SET {','.join(fields)} WHERE id=?",values)
    def add_context_segment(self, conversation_id: str, chunk_id: str, role: str, text: str, token_count: int):
        sequence=self.conn.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM context_segments WHERE conversation_id=?",(conversation_id,)).fetchone()[0]
        segment_id=str(uuid.uuid4())
        with self.conn:self.conn.execute("INSERT INTO context_segments VALUES (?,?,?,?,?,?,?,?,?)",(segment_id,conversation_id,chunk_id,sequence,role,text,token_count,1,self._now()))
        return segment_id
    def active_context_records(self, conversation_id: str | None = None):
        sql="SELECT s.*,v.offset,v.id vector_id FROM context_segments s JOIN vectors v ON v.chunk_id=s.chunk_id WHERE s.active=1"; params=[]
        if conversation_id:sql+=" AND s.conversation_id=?";params.append(conversation_id)
        return [dict(row) for row in self.conn.execute(sql,params)]
    def unsynced_context_records(self, conversation_id: str):
        row=self.conn.execute("SELECT last_sequence FROM conversation_syncs WHERE conversation_id=?",(conversation_id,)).fetchone(); last=row[0] if row else 0
        return [dict(row) for row in self.conn.execute("SELECT * FROM context_segments WHERE conversation_id=? AND active=1 AND sequence>? ORDER BY sequence",(conversation_id,last))]
    def conversation_sync_status(self, conversation_id: str):
        row=self.conn.execute("SELECT * FROM conversation_syncs WHERE conversation_id=?",(conversation_id,)).fetchone()
        return dict(row) if row else {"conversation_id":conversation_id,"last_sequence":0,"status":"idle"}
    def mark_conversation_synced(self, conversation_id: str, sequence: int, status: str = "complete"):
        with self.conn:self.conn.execute("INSERT INTO conversation_syncs (conversation_id,last_sequence,status,updated_at) VALUES (?,?,?,?) ON CONFLICT(conversation_id) DO UPDATE SET last_sequence=excluded.last_sequence,status=excluded.status,updated_at=excluded.updated_at",(conversation_id,sequence,status,self._now()))
    def recent_context_records(self, conversation_id: str, limit: int = 4):
        rows=self.conn.execute("SELECT s.*,v.offset,v.id vector_id FROM context_segments s JOIN vectors v ON v.chunk_id=s.chunk_id WHERE s.active=1 AND s.conversation_id=? ORDER BY s.sequence DESC LIMIT ?",(conversation_id,limit))
        return list(reversed([dict(row) for row in rows]))
    def archive_context_segments(self, segment_ids: list[str]):
        if not segment_ids:return
        p=','.join('?'*len(segment_ids))
        with self.conn:self.conn.execute(f"UPDATE context_segments SET active=0 WHERE id IN ({p})",segment_ids)
    def create_operation_trace(self, conversation_id: str | None, operation_type: str):
        trace_id=str(uuid.uuid4()); now=self._now()
        with self.conn:self.conn.execute("INSERT INTO operation_traces VALUES (?,?,?,?,?,?,?)",(trace_id,conversation_id,operation_type,"running","[]",now,now))
        return trace_id
    def append_operation_trace(self, trace_id: str, event_type: str, payload: dict):
        row=self.conn.execute("SELECT events FROM operation_traces WHERE id=?",(trace_id,)).fetchone()
        if not row:return
        events=json.loads(row[0]); events.append({"event":event_type,"at":self._now(),"payload":payload})
        with self.conn:self.conn.execute("UPDATE operation_traces SET events=?,updated_at=? WHERE id=?",(json.dumps(events),self._now(),trace_id))
    def complete_operation_trace(self, trace_id: str, status: str):
        with self.conn:
            self.conn.execute("UPDATE operation_traces SET status=?,updated_at=? WHERE id=?",(status,self._now(),trace_id))
            self.conn.execute("DELETE FROM operation_traces WHERE id IN (SELECT id FROM operation_traces ORDER BY created_at DESC LIMIT -1 OFFSET 200)")
    def recent_operation_traces(self, limit: int = 50):
        rows=self.conn.execute("SELECT * FROM operation_traces ORDER BY created_at DESC LIMIT ?",(limit,))
        return [{**dict(row),"events":json.loads(row["events"])} for row in rows]
    def operation_trace(self, trace_id: str):
        row=self.conn.execute("SELECT * FROM operation_traces WHERE id=?",(trace_id,)).fetchone()
        return {**dict(row),"events":json.loads(row["events"])} if row else None
    def personal_node_details(self, collection_id: str, node_id: str):
        node=self.personal_node(node_id)
        if not node:return None
        children=self.personal_children(collection_id,node_id)
        rows=self.conn.execute("SELECT * FROM personal_items WHERE collection_id=? AND node_id=? ORDER BY updated_at DESC",(collection_id,node_id))
        items=[dict(row) for row in rows]
        pending=[dict(row) for row in self.conn.execute("SELECT * FROM personal_updates WHERE collection_id=? AND proposed_node_id=? AND status='pending' ORDER BY created_at DESC",(collection_id,node_id))]
        return {"node":node,"children":children,"current_facts":[item for item in items if item["status"]=="current"],"history":[item for item in items if item["status"]=="superseded"],"pending_updates":pending,"direct_fact_count":len(items)}
