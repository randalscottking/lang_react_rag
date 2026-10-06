import logging
import threading

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

log = logging.getLogger("rag")

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions using only the "
            "context provided below. If the context does not contain the answer, "
            "say you don't know based on the available documents. Be concise.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


class LocalEmbeddings(Embeddings):
    """LangChain wrapper around Chroma's bundled ONNX MiniLM embedder.

    Runs locally (no torch, no extra API key). Anthropic does not offer an
    embeddings endpoint, so embeddings are computed in-process.
    """

    def __init__(self) -> None:
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs)


class RagService:
    def __init__(self) -> None:
        self._embeddings = LocalEmbeddings()
        self._lock = threading.Lock()
        self._store = self._new_store()
        self._chain = None

    def _new_store(self) -> Chroma:
        return Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=config.CHROMA_DIR,
        )

    def _load_documents(self) -> list[Document]:
        docs: list[Document] = []
        root = config.DOCS_DIR
        if not root.exists():
            log.warning("Documents directory %s does not exist", root)
            return docs
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in config.SUPPORTED_EXTENSIONS:
                try:
                    loaded = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
                except Exception:
                    log.exception("Failed to load %s", path)
                    continue
                for d in loaded:
                    d.metadata["source"] = str(path.relative_to(root))
                docs.extend(loaded)
        return docs

    def ingest(self) -> dict:
        """Rebuild the vector index from the documents directory."""
        with self._lock:
            docs = self._load_documents()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(docs)

            try:
                self._store.delete_collection()
            except Exception:
                log.debug("No existing collection to delete", exc_info=True)
            self._store = self._new_store()

            if chunks:
                self._store.add_documents(chunks)
            log.info("Indexed %d files into %d chunks", len(docs), len(chunks))
            return {"files": len(docs), "chunks": len(chunks)}

    def _get_chain(self):
        if self._chain is None:
            if not config.ANTHROPIC_API_KEY:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            llm = ChatAnthropic(
                model=config.CLAUDE_MODEL,
                api_key=config.ANTHROPIC_API_KEY,
                temperature=0,
                max_tokens=1024,
            )
            self._chain = PROMPT | llm | StrOutputParser()
        return self._chain

    def ask(self, question: str) -> dict:
        chain = self._get_chain()
        docs = self._store.similarity_search(question, k=config.TOP_K)
        if not docs:
            return {
                "answer": "No documents are indexed yet. Add .txt or .md files to the documents folder and reindex.",
                "sources": [],
            }
        answer = chain.invoke({"context": format_docs(docs), "question": question})
        sources = sorted({d.metadata.get("source", "unknown") for d in docs})
        return {"answer": answer, "sources": sources}
