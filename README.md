# lang_react_rag

Minimal RAG app: LangChain + Chroma + Claude, served by FastAPI, with a one-box React front end.

## Architecture

![Architecture diagram](docs/architecture.svg)

```mermaid
flowchart LR
    User([User]) --> UI["React UI<br/>(served by nginx, :3000)"]
    UI -- "POST /api/chat" --> API["FastAPI backend<br/>(:8000)"]

    subgraph Indexing["Indexing (startup and /api/reindex)"]
        direction TB
        Docs[("Documents folder<br/>~/Documents/RAGapp")] --> Loader["LangChain loader<br/>+ text splitter"]
        Loader --> Embed["Local embeddings<br/>(MiniLM, ONNX)"]
        Embed --> Chroma[("Chroma<br/>vector store")]
    end

    subgraph Query["Query time"]
        direction TB
        Retriever["LangChain retriever"] --> Prompt["Prompt: context + question"]
        Prompt --> Claude["Claude API<br/>(Anthropic)"]
    end

    API --> Retriever
    Retriever -. "embed question" .-> Embed
    Retriever -- "top-k similarity search" --> Chroma
    Claude -- "answer + sources" --> API
    API --> UI
    API -. "startup / reindex" .-> Loader
```

**Query flow:** the question goes from the browser through nginx to FastAPI. The question is embedded locally and Chroma returns the closest chunks. Those chunks and the question are sent to Claude, which answers from the retrieved context only. The answer and source file names return to the UI.

**Indexing flow:** on startup (and on `POST /api/reindex`) the backend reads `.txt` / `.md` files, splits them into chunks, embeds each chunk with the local model, and writes the vectors to Chroma.

Claude is used only for generation. Embeddings come from a separate local model, because Anthropic does not provide an embeddings endpoint.

## How it works

- On startup the backend reads every `.txt` / `.md` file under the documents folder, splits it into chunks, embeds it locally (Chroma's bundled MiniLM model), and stores it in Chroma.
- `POST /api/chat` retrieves the top matching chunks and asks Claude to answer from them only.
- `POST /api/reindex` rebuilds the index after you add or change files.

## Guardrails

Implemented in `backend/app/guardrails.py`, all within LangChain (no external guardrail service):

1. **Regex pre-filter** blocks obvious prompt-injection phrasing.
2. **PII redaction** (regex: email, SSN, Luhn-checked cards, phone, IPv4) on the question before it reaches the model, and on the answer before it is returned.
3. **Input guard**: a small Claude model (`GUARD_MODEL`) classifies the question via structured output (prompt attacks, off-topic, harmful).
4. **Prompt hardening**: retrieved chunks are wrapped in `<document>` tags and treated as data, not instructions.
5. **Grounding check**: a judge model verifies the answer is supported by the retrieved chunks; otherwise it is replaced and sources are withheld.

LLM-backed checks fail closed. Run tests with `pip install -r backend/requirements-dev.txt && cd backend && pytest`.

## Run with Docker

```bash
cp .env.example .env        # then set ANTHROPIC_API_KEY
docker compose up --build
```

Open http://localhost:3000. By default the app indexes `~/Documents/RAGapp`; set `DOCS_DIR_HOST` in `.env` to use another folder.

Reindex after changing documents:

```bash
curl -X POST http://localhost:8000/api/reindex
```

## Local development

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=... DOCS_DIR=$HOME/Documents/RAGapp CHROMA_DIR=./chroma
uvicorn app.main:app --reload

# frontend (separate terminal)
cd frontend
npm install
npm run dev                 # http://localhost:5173, proxies /api to :8000
```

## Configuration

| Variable | Default |
| --- | --- |
| `ANTHROPIC_API_KEY` | required |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` |
| `TOP_K` | `4` |
| `GUARDRAILS_ENABLED` | `true` |
| `GUARD_MODEL` | `claude-haiku-5-5` |
| `ALLOWED_TOPICS` | `questions about the contents of the indexed documents` |
