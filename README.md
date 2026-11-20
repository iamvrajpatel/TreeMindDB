# TreeMindDB

TreeMindDB is a localhost-first personal assistant built on a custom tree-based vector database. It stores structured personal knowledge in SQLite and normalized vectors in local binary segments; it does not use an external vector database.

## Run locally

```bash
cd backend
../.venv/bin/pip install -e '.[test]'
TREEMIND_DATA_DIR=../data ../.venv/bin/uvicorn app.main:app --reload
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Start Ollama with configured `OLLAMA_MODEL` and `OLLAMA_EMBEDDING_MODEL=nomic-embed-text`, then open `http://localhost:3000`. The app creates a persistent `My Personal Knowledge` root with Profile, Projects, Experience & Skills, Certifications, Relationships, and Preferences branches. Every message or uploaded source is retained as provenance, then explicit durable facts are extracted and classified into existing or new branches.

## Retrieval and model status

Production personal-memory retrieval uses Ollama's dedicated `nomic-embed-text` model. The deterministic provider remains only for tests and degraded first-run diagnostics; `/api/v1/models/status` makes that state explicit.

Active context and the personal tree are searched in parallel on every chat turn. Laya chooses semantic personal-tree branches and the selected nodes are highlighted beside the chat. If Laya cannot run or returns invalid node IDs, chat visibly uses local semantic branch ranking; its fallback reason is persisted in the trace. Laya decision scores are never reported as cosine similarity.

Ollama writes replies, curates durable personal facts, and summarizes tree nodes. When a new fact conflicts with a current fact, TreeMindDB asks for confirmation before replacing it. If Ollama is not configured, the UI reports that status and no knowledge is created.

## Storage and operations

Every upload appends normalized float32 vectors to `data/vectors/<collection>.vectors` and inserts SQLite mappings. Original source text is retained separately from atomic facts, and each fact has source provenance visible in node inspection. The binary store writes a staging segment before append; startup SQLite metadata is the source of truth. Do not mix embedding models in one collection; re-embed from retained sources before changing model or dimension.

Run backend tests with `cd backend && PYTHONPATH=. ../.venv/bin/pytest -q`.
