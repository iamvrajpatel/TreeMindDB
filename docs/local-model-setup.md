# Run Ollama and Laya Locally

TreeMindDB uses two local model systems:

- **Ollama** generates replies, extracts compact memory notes, and labels tree clusters.
- **Laya** makes typed branch decisions while TreeMindDB retrieves memory from the semantic tree. It does not generate chat responses.

The following instructions target Linux and this project directory:

```bash
cd /home/vrajpatel/personal/TreeMind
```

## 1. Install and run Ollama

Install Ollama with its official Linux installer:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start its local server in a separate terminal if it is not already managed as a service:

```bash
ollama serve
```

In another terminal, download a small general-purpose model. `llama3.2:3b` is a reasonable starting point for local chat, summaries, and short tree labels:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama run llama3.2:3b "Reply with: Ollama is ready."
```

Check installed models at any time:

```bash
ollama list
```

If the Ollama server is running on a different host or port, set `OLLAMA_BASE_URL` accordingly; otherwise TreeMindDB uses `http://localhost:11434`.

## 2. Install Laya into TreeMindDB's virtual environment

Laya runs in the backend Python process. Install it with the same interpreter that starts FastAPI:

```bash
cd /home/vrajpatel/personal/TreeMind
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --upgrade laya torch
```

The first Laya request downloads its Hugging Face checkpoint(s). TreeMindDB currently creates `Router(preload=True)`, so its first backend startup may download and load all configured Laya checkpoints. This can take several minutes and requires internet access, disk space, and considerably more RAM than the rest of the app.

Use this smoke test before starting TreeMindDB:

```bash
.venv/bin/python - <<'PY'
from laya import Router

router = Router()
result = router.predict(
    {"query": "How do I retrieve an earlier database preference?"},
    {
        "branch": {
            "type": "choice",
            "instructions": "Select the relevant memory branch.",
            "criteria": {
                "database": "database architecture and retrieval preferences",
                "personal": "personal preferences and profile information",
            },
        }
    },
)
print(result["answers"]["branch"])
print(result["routing"])
PY
```

For a CPU-only machine, Laya can be created with `device="cpu"`; it will be slower. GPU-capable PyTorch installations should follow the PyTorch selector for the correct CUDA/driver combination before installing Laya.

## 3. Configure TreeMindDB

Copy the project environment template and set the local Ollama model name:

```bash
cd /home/vrajpatel/personal/TreeMind
cp .env.example .env
```

Set these values in `.env`:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
TREEMIND_DATA_DIR=./data
CORS_ORIGINS=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

Load the values before starting the backend:

```bash
set -a
. ./.env
set +a
cd backend
../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

In a second terminal, start the frontend:

```bash
cd /home/vrajpatel/personal/TreeMind/frontend
npm run dev
```

Open `http://localhost:3000`.

## 4. Verify TreeMindDB model status

After the backend is running, inspect its model status:

```bash
curl http://127.0.0.1:8000/api/v1/models/status
curl http://127.0.0.1:8000/api/v1/chat-memory/status
```

Expected conditions:

- `generation.configured` is `true` when `OLLAMA_MODEL` is set.
- `memory_agent_available` is `true` when the local generation provider is configured.
- `laya.available` is `true` after Laya imports and initializes correctly.
- `embedding.mode` is `semantic` when `nomic-embed-text` is available.

When Laya is unavailable or returns an invalid branch, TreeMindDB exposes a local semantic-routing fallback and its reason in the UI trace. When Ollama generation is unavailable, TreeMindDB retains the source but defers fact extraction.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `ollama: command not found` | Re-run the Ollama installer, then open a new shell. |
| `connection refused` on port 11434 | Start `ollama serve`, or start/enable the Ollama system service if your installation supplies one. |
| `Laya unavailable: ModuleNotFoundError` | Run `.venv/bin/python -m pip install -U laya torch`; make sure Uvicorn is started with `.venv/bin/python` or `.venv/bin/uvicorn`. |
| First Laya request is slow | Normal on first checkpoint download/load. Keep the model cache on local disk and allow the initial preload to finish. |
| Out-of-memory while loading Laya | Use a machine with more RAM/VRAM, configure the router for CPU/one resident model in code, or run the Laya HTTP service separately. |
| Tree shows Laya fallback | Check `/api/v1/models/status`, then inspect backend logs for the Laya load or decision error. |

## References

- [Ollama installation](https://docs.ollama.com/linux)
- [Laya Quickstart and Router](https://huggingface.co/convaiinnovations/laya)
- [Laya local setup](https://github.com/NandhaKishorM/laya/blob/main/LOCAL_SETUP.md)
- [Laya router loading and memory behavior](https://github.com/NandhaKishorM/laya/blob/main/docs/routing.md)

