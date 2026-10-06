import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config
from .rag import RagService

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

rag = RagService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.ANTHROPIC_API_KEY:
        log.warning("ANTHROPIC_API_KEY is not set; /api/chat will return 503")
    stats = rag.ingest()
    log.info("Startup ingest complete: %s", stats)
    yield


app = FastAPI(title="lang_react_rag", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return rag.ask(req.question.strip())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        log.exception("Chat request failed")
        raise HTTPException(status_code=500, detail="Something went wrong answering that question.")


@app.post("/api/reindex")
def reindex():
    return rag.ingest()
