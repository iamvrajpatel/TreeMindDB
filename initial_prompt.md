# Build TreeMindDB — An Agentic Tree-Based Vector Database with Laya-Guided Retrieval

You are a senior database engineer, machine learning engineer, backend architect, and frontend developer.

Build a complete, locally runnable, full-stack application called **TreeMindDB**.

The project must implement a custom tree-based vector database from scratch, integrate the Laya decision model for intelligent hierarchical retrieval, provide an AI agent that interacts with the database, and include a modern web dashboard for visualizing the tree, exploring stored information, monitoring retrieval, and chatting with indexed documents.

This must be a functional application, not a static prototype or mock dashboard.

## 1. Core Objectives

Implement the following:

1. A persistent tree-based vector database.
2. Document ingestion and embedding generation.
3. Hierarchical semantic clustering.
4. Laya-powered decision-based tree navigation.
5. Multiple retrieval strategies for comparison.
6. A retrieval-augmented generation (RAG) agent.
7. A chatbot that can query stored information.
8. An interactive graph/tree visualization.
9. A node inspector showing exactly what is stored in each node.
10. A retrieval debugger showing the model's routing decisions.
11. An evaluation dashboard for retrieval accuracy and latency.
12. Complete REST APIs, tests, documentation, and Docker setup.

## 2. Technology Stack

### Backend
- Python 3.11+
- FastAPI
- Pydantic
- NumPy
- Sentence Transformers
- PyTorch where required
- Laya SDK
- SQLite for metadata
- Local binary files for vector persistence
- FAISS optional for baseline benchmarking only
- Uvicorn
- pytest

### Frontend
- Next.js with TypeScript
- Tailwind CSS
- shadcn/ui
- React Flow for interactive tree visualization
- Recharts for analytics
- Lucide icons
- TanStack Query for API communication
- Zustand for UI state
- WebSocket or Server-Sent Events for streaming retrieval events

### AI
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` by default.
- Decision model: `convaiinnovations/laya`.
- Generative LLM: provider-independent interface supporting Ollama and an OpenAI-compatible API.

Laya must not be used as the generative chatbot model. It is a decision/routing component.

Use a clean, modular architecture that allows changing embedding, routing, and generation models independently.

## 3. Tree-Based VectorDB Engine

Build a custom hierarchical vector indexing engine.

Do not use Pinecone, Weaviate, Chroma, Qdrant, Milvus, or another existing vector database as the primary storage engine.

The application must own its vector indexing and retrieval implementation.

### Tree Structure

Each internal tree node should store:

- Unique node identifier
- Parent identifier
- Child identifiers
- Tree depth
- Node type
- Cluster centroid
- Vector count
- Document count
- Cluster radius or dispersion statistics
- Human-readable semantic summary
- Optional semantic label
- Creation timestamp
- Updated timestamp
- Cluster version

Each leaf node should additionally reference:

- Vector identifiers
- Document chunk identifiers
- Embedding storage locations
- Source document identifiers

Do not store entire large embedding arrays directly in frontend visualization responses.

Store actual embeddings in efficient contiguous numerical storage.

### Tree Construction

Implement hierarchical K-means or another documented clustering approach.

Configuration parameters:

- `branching_factor`
- `max_leaf_size`
- `max_tree_depth`
- `minimum_split_size`
- `distance_metric`
- `embedding_dimension`

Default values:

- Branching factor: 4
- Maximum leaf size: 128
- Maximum depth: 8
- Distance metric: cosine similarity

These defaults must be configurable.

Tree construction algorithm:

1. Receive normalized document embeddings.
2. Create a root node containing all vectors.
3. Calculate the centroid and statistics.
4. Split oversized nodes using hierarchical clustering.
5. Recursively create child nodes.
6. Stop when leaf size or depth limits are reached.
7. Persist nodes, vectors, and memberships.
8. Generate semantic descriptions for internal nodes.
9. Validate tree integrity.

For cosine clustering, use normalized vectors and a suitable spherical or normalized clustering approach.

Handle empty clusters, duplicate vectors, and degenerate partitions.

### Insert, Update, and Delete

Implement:

- Insert document
- Insert vectors
- Batch insert
- Update document
- Delete document
- Delete vector
- Rebuild tree
- Rebalance tree
- Retrieve by vector ID
- Retrieve by document ID

Dynamic insertion must preserve a valid index. Implement a clearly defined insertion policy using nearest-centroid assignment and an explicit split/rebuild policy.

Document all limitations of incremental updates.

### Persistence

Implement durable local persistence:

- SQLite stores document metadata, chunks, node records, and vector mappings.
- Binary or memory-mapped storage stores embeddings.
- Tree adjacency and centroid data must persist across application restarts.
- Use atomic index snapshots or a write-ahead/recovery mechanism.
- Store embedding model identity and embedding dimension with each collection.
- Detect incompatible models and reject invalid inserts.
- Recover safely from interrupted writes.
- Prevent orphaned vector references after deletion.

Implement startup recovery and index validation.

## 4. Laya Decision-Based Retrieval Engine

Integrate the model:

https://huggingface.co/convaiinnovations/laya

Study the official documentation before implementing the integration.

Use the documented Laya SDK and its typed decision interfaces.

Do not invent unsupported Laya API methods.

### Laya's Responsibility

Laya must guide the search through the hierarchical tree.

For a user query:

1. Generate a query embedding.
2. Load candidate child nodes from the current tree node.
3. Prepare a decision context containing the query and child-node descriptions.
4. Ask Laya to select or score the most relevant child branches.
5. Retain multiple promising branches.
6. Recursively navigate the tree.
7. Search selected leaf nodes using actual vector similarity.
8. Merge candidate results.
9. Return the top-k matches.

Do not treat Laya confidence as cosine similarity.

Do not assume Laya's default confidence values are calibrated for nearest-neighbor containment.

### Decision Context

Each routing decision must receive:

- User query
- Current node summary
- Candidate child IDs
- Candidate child semantic labels
- Candidate child summaries
- Optional cluster statistics
- Routing instructions

The routing output should include:

- Selected child identifiers
- Decision scores when supported
- Inference latency
- Model identifier
- Routing depth
- Fallback status

Create a strongly typed adapter around Laya rather than coupling the search engine directly to SDK response dictionaries.

### Routing Example

Use the official Laya typed-choice pattern conceptually:

```python
from laya import Router

router = Router()

questions = {
    "branch": {
        "type": "choice",
        "instructions": (
            "Select the tree branch most relevant "
            "to retrieving the query."
        ),
        "criteria": {
            "child_1": "Semantic description of child 1",
            "child_2": "Semantic description of child 2",
            "child_3": "Semantic description of child 3",
        },
    }
}

result = router.predict(query, questions)
```

Verify the installed SDK's supported input shape and response schema.

Laya should operate through an interface such as:

```python
class DecisionRouter:
    def rank_children(
        self,
        query: str,
        children: list,
    ) -> list:
        ...
```

If a true ranking distribution is unavailable, do not fabricate one.

Implement alternative multi-branch candidate generation using independently evaluated decisions or geometric centroid ranking.

### Multi-Branch Search

Implement beam search.

Configuration:

- Beam width
- Maximum tree depth
- Maximum visited nodes
- Maximum candidate vectors
- Routing confidence threshold
- Retrieval timeout

Never assume that routing to one leaf guarantees the nearest neighbor.

Implement multi-probe retrieval to reduce partition errors.

### Retrieval Modes

Expose four selectable modes:

**Mode A: Exact Search**

Search all stored vectors and calculate the true nearest neighbors.

**Mode B: Centroid Tree Search**

Use query-to-centroid similarity to select promising tree branches.

**Mode C: Laya-Guided Tree Search**

Use Laya to guide hierarchical traversal.

**Mode D: Hybrid Search**

Combine Laya decisions and geometric centroid distances to choose candidate branches.

Optionally add HNSW inside leaves later as a separate extension.

All retrieval modes must use the same embedding model and distance metric so comparisons are meaningful.

### Fallback Logic

When Laya is unavailable, fails, times out, or returns invalid child IDs:

- Log the failure.
- Switch to centroid-based routing.
- Preserve a usable retrieval experience.
- Show fallback status in the frontend.
- Never silently invent routing confidence scores.

## 5. Document Ingestion Pipeline

Allow ingestion of:

- PDF
- TXT
- Markdown
- CSV
- JSON
- Plain text entered in the UI

Pipeline:

1. Upload document.
2. Extract text.
3. Normalize text.
4. Split into configurable chunks.
5. Preserve document and page references.
6. Generate embeddings.
7. Store embeddings.
8. Update tree index.
9. Update metadata.
10. Return ingestion statistics.

For every chunk, store:

- Chunk ID
- Document ID
- Chunk text
- Source name
- Page number, where available
- Chunk position
- Token count
- Embedding ID
- Metadata
- Timestamp

Support duplicate detection and clear ingestion error reporting.

## 6. Agentic Retrieval System

Build an agent that can interact with TreeMindDB through explicit tools.

The agent must be able to:

- Search indexed documents.
- Inspect the tree structure.
- Retrieve node metadata.
- Compare retrieval strategies.
- Retrieve full source chunks.
- Explain a search path.
- Answer questions using retrieved context.
- Report missing evidence.
- Summarize indexed documents.

Implement tools with typed input/output contracts:

```python
search_vectors(query, top_k, mode)
inspect_tree()
get_node_details(node_id)
get_document(document_id)
get_chunk(chunk_id)
compare_search_modes(query)
explain_retrieval(query)
```

The agent must not directly modify database internals.

Document ingestion and deletion must use separately authorized API operations.

### Agent Workflow

User message
→ Intent detection
→ Tool selection
→ Tree retrieval
→ Source chunk collection
→ Context assembly
→ LLM response
→ Citations and retrieval trace

Use Laya for decision-based routing and optionally intent classification.

Use the generative LLM for reasoning over retrieved content and producing natural language responses.

The agent should clearly distinguish source-supported information from assumptions.

## 7. RAG Chatbot

Build a complete chatbot interface connected to the backend.

Features:

- Natural language document questions
- Streaming assistant responses
- Conversation history
- Configurable top-k
- Retrieval mode selector
- Source citations
- Expandable retrieved context
- Document links
- Similarity scores
- Routing trace
- Token usage where available
- Latency details
- Model status display

Each assistant response should expose an expandable retrieval explanation.

Example:

User:
"How does HNSW work?"

Backend:
1. Embeds the query.
2. Selects candidate tree branches.
3. Searches vectors in selected leaves.
4. Retrieves relevant chunks.
5. Passes context to the LLM.
6. Produces an answer with citations.

The UI should display which tree nodes were visited during retrieval.

When a user clicks a citation, open the corresponding source document or chunk.

If no relevant evidence is found, the chatbot must say so rather than fabricate an answer.

## 8. Frontend Dashboard

Build a modern, professional developer-tool interface.

Use a dark-first design with a light-mode option.

Suggested layout:

**Left sidebar**
- Overview
- Tree Explorer
- Documents
- Vector Search
- Chatbot
- Retrieval Debugger
- Benchmarks
- Settings

### Page A: Overview

Display:

- Total documents
- Total vectors
- Total chunks
- Total tree nodes
- Leaf nodes
- Tree depth
- Embedding model
- Laya model status
- Storage size
- Index health
- Recent ingestion operations

Include useful charts showing vector distribution by depth and cluster.

### Page B: Interactive Tree Explorer

This is the most important frontend feature.

Use React Flow to visualize the actual stored tree.

Requirements:

- Zoom
- Pan
- Expand/collapse nodes
- Search nodes by ID or label
- Highlight parent-child relationships
- Highlight selected node
- Display number of stored vectors
- Display cluster sizes
- Show cluster hierarchy
- Show selected retrieval paths
- Load large trees lazily

Do not render thousands of nodes simultaneously.

### Tree Node Cards

Each tree node should show:

- Node name or ID
- Node type
- Depth
- Number of vectors
- Number of documents
- Semantic label
- Parent-child relationships

Use different visual styles for:

- Root nodes
- Internal nodes
- Leaf nodes
- Currently visited nodes
- Selected nodes
- Skipped nodes

### Node Inspector

When a node is clicked, open a right-side detail panel.

Display:

**Node Information**
- ID
- Type
- Depth
- Parent ID
- Child IDs
- Vector count
- Cluster centroid summary
- Cluster radius
- Semantic label
- Semantic description

**Stored Information**
- Referenced documents
- Stored chunks
- Chunk previews
- Source metadata
- Embedding IDs
- Vector dimension
- Vector value preview

**Actions**
- Expand children
- Search within subtree
- View representative vectors
- Copy node ID
- Inspect source document
- Compare child clusters

Explicitly distinguish storage references from actual vector payloads.

The frontend must make it obvious exactly what information is stored at each tree level.

### Page C: Documents

Provide:

- Drag-and-drop upload
- Upload progress
- Document list
- Source metadata
- Chunk viewer
- Vector count per document
- Tree placement
- Document delete operation
- Re-index action

### Page D: Vector Search Playground

Allow users to enter a query and select:

- Exact search
- Tree search
- Laya search
- Hybrid search

Show:

- Retrieved results
- Similarity scores
- Selected tree branches
- Visited node count
- Candidate vector count
- Total latency
- Laya inference latency
- Search latency

### Page E: Retrieval Debugger

Visualize an individual retrieval request.

Example:

Query:
"How do vector databases use HNSW?"

Trace:

Root
→ Machine Learning
→ Vector Search
→ ANN Algorithms
→ Leaf 42

Show:

- Decision steps
- Branch candidates
- Laya output
- Centroid similarity
- Chosen branch
- Skipped branches
- Beam width
- Leaf search results
- Final ranking

Animate the path through the actual tree using trace events.

Every displayed decision must originate from a real backend trace.

### Page F: Chatbot

Create a split-screen layout:

Left:
- Chat messages
- Input box
- Conversation history
- Retrieval settings

Right:
- Live tree visualization
- Active routing path
- Retrieved chunks
- Source citations
- Latency breakdown

When a user sends a message, animate the traversal as actual retrieval events arrive from the backend.

Do not simulate retrieval animations with hardcoded paths.

### Page G: Benchmark Dashboard

Compare:

- Exact search
- Centroid tree
- Laya tree
- Hybrid tree

Show:

- Recall@1
- Recall@5
- Recall@10
- p50 latency
- p95 latency
- Queries per second
- Visited nodes
- Candidate vectors
- Memory usage

Recall must be calculated against exact nearest-neighbor ground truth.

Include charts comparing recall and latency as beam width changes.

Display benchmark dataset size, hardware, index configuration, and model versions.

## 9. Backend API

Implement versioned REST endpoints.

### Documents

```text
POST   /api/v1/documents/upload
GET    /api/v1/documents
GET    /api/v1/documents/{id}
DELETE /api/v1/documents/{id}
GET    /api/v1/documents/{id}/chunks
```

### Tree

```text
GET    /api/v1/tree
GET    /api/v1/tree/stats
GET    /api/v1/tree/nodes/{id}
GET    /api/v1/tree/nodes/{id}/children
GET    /api/v1/tree/nodes/{id}/vectors
POST   /api/v1/tree/rebuild
GET    /api/v1/tree/health
```

### Retrieval

```text
POST   /api/v1/search
POST   /api/v1/search/compare
POST   /api/v1/search/explain
GET    /api/v1/search/traces/{id}
```

### Chat

```text
POST   /api/v1/chat
POST   /api/v1/chat/stream
GET    /api/v1/chat/conversations
GET    /api/v1/chat/conversations/{id}
```

### Evaluation

```text
POST   /api/v1/benchmarks/run
GET    /api/v1/benchmarks
GET    /api/v1/benchmarks/{id}
```

### System

```text
GET    /api/v1/health
GET    /api/v1/models/status
GET    /api/v1/config
```

Use Pydantic request/response schemas.

Add pagination for large document, vector, and tree responses.

Support structured errors and request tracing.

## 10. Project Directory Structure

Create a monorepo organized approximately as follows:

```text
treeminddb/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── vectordb/
│   │   │   ├── tree.py
│   │   │   ├── clustering.py
│   │   │   ├── storage.py
│   │   │   ├── indexing.py
│   │   │   ├── search.py
│   │   │   └── persistence.py
│   │   ├── routing/
│   │   │   ├── base.py
│   │   │   ├── laya_router.py
│   │   │   ├── centroid_router.py
│   │   │   └── hybrid_router.py
│   │   ├── ingestion/
│   │   ├── embeddings/
│   │   ├── agents/
│   │   ├── rag/
│   │   ├── benchmarks/
│   │   ├── schemas/
│   │   └── main.py
│   ├── tests/
│   ├── pyproject.toml
│   └── Dockerfile
├── frontend/
│   ├── app/
│   ├── components/
│   │   ├── tree/
│   │   ├── chat/
│   │   ├── documents/
│   │   ├── search/
│   │   └── dashboards/
│   ├── hooks/
│   ├── lib/
│   ├── types/
│   └── package.json
├── data/
├── scripts/
├── benchmarks/
├── docker-compose.yml
├── .env.example
└── README.md
```

## 11. Testing Requirements

Write automated tests for:

- Tree creation
- Correct parent-child relationships
- Vector insertion
- Vector deletion
- Persistence after restart
- Centroid computation
- Similarity search
- Exact ground-truth retrieval
- Multi-branch search
- Laya output parsing
- Invalid routing decisions
- Fallback behavior
- API validation
- Document ingestion
- Agent tool execution
- Chat source citations

Mock external model dependencies in unit tests.

Create integration tests using a small deterministic embedding dataset.

Test a realistic end-to-end workflow:

1. Start the application.
2. Upload documents.
3. Generate vectors.
4. Build the tree.
5. Inspect tree nodes.
6. Query the database.
7. View retrieval traces.
8. Ask questions through the chatbot.
9. Restart the backend.
10. Verify data is still available.

## 12. Security and Reliability

Implement:

- File type validation
- Upload size limits
- Safe PDF parsing
- CORS configuration
- Request validation
- Safe file paths
- Environment-based secrets
- Structured logging
- Controlled agent tool access
- Safe error handling

Treat uploaded documents as untrusted content.

Retrieved document text must not be allowed to override the agent's tool permissions or system instructions.

Do not expose internal server paths or credentials through the frontend.

## 13. Development Phases

Build in the following order.

### Phase 1: Core Database
Implement vector storage, hierarchy construction, exact search, centroid-based tree search, persistence, and tests.

### Phase 2: Laya Integration
Implement the typed decision router, routing traces, multi-branch traversal, and fallback policies.

### Phase 3: Agent and Chat
Implement retrieval tools, RAG orchestration, source citations, and chat APIs.

### Phase 4: Frontend
Build the dashboard, tree explorer, node inspector, documents page, and chatbot.

### Phase 5: Retrieval Visualization
Connect live backend traces to the React Flow tree and animate actual search paths.

### Phase 6: Evaluation
Implement reproducible retrieval benchmarks and compare all retrieval modes.

### Phase 7: Deployment
Add Docker, environment configuration, startup instructions, sample data, and system diagnostics.

## 14. Mandatory Engineering Rules

- Do not substitute a regular vector database for the custom tree engine.
- Do not fake Laya integration.
- Do not generate artificial retrieval results for the real search UI.
- Do not use Laya for text generation.
- Do not claim approximate search is exact.
- Do not treat Laya routing confidence as vector similarity.
- Do not assume one selected branch contains all nearest neighbors.
- Do not hardcode tree visualization data.
- Do not use random numbers for benchmark charts.
- Do not silently swallow model or persistence errors.
- Do not leave the frontend disconnected from the backend.
- Do not claim improved retrieval performance without benchmarks.
- Keep components modular and individually testable.
- Provide a deterministic demo mode backed by a real small dataset.
- Clearly label disabled functionality if model dependencies are unavailable.

## 15. Definition of Done

The project is complete when I can:

1. Start the backend and frontend locally.
2. Upload a PDF.
3. Watch it get chunked and embedded.
4. See its vectors assigned to the tree.
5. Open the interactive tree visualization.
6. Click any node to inspect what it contains.
7. Search the database using Laya-guided retrieval.
8. Observe real branch-selection decisions.
9. Compare Laya search against centroid and exact search.
10. Ask the chatbot questions about uploaded documents.
11. View citations and the retrieval path for every answer.
12. Restart the application without losing indexed data.
13. Run tests and retrieval benchmarks.
14. Inspect meaningful performance measurements.

## 16. Final Deliverables

Produce:

- Complete backend source code
- Complete frontend source code
- Database schema
- Vector storage implementation
- Tree indexing engine
- Laya adapter
- RAG agent
- Chatbot
- Interactive tree explorer
- Retrieval debugger
- Benchmark system
- Unit and integration tests
- Docker configuration
- Example documents
- README with setup instructions
- Architecture documentation

Before writing code, define the architecture, storage contracts, and API schemas.

Then implement the project phase by phase.

At every phase, run available tests and fix failures before continuing.

Prefer a genuinely working MVP over a broad but nonfunctional scaffold.

Begin with the project structure, the core database engine, and a minimal vertical slice that demonstrates:

Document → Embedding → Tree Index → Laya Routing → Vector Retrieval → Chat Response → Visual Retrieval Trace.
