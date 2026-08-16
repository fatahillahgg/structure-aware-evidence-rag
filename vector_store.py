from pathlib import Path
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from chunker import chunk_documents
from embedder import create_embedder
from loader import load_paper


DEFAULT_INDEX_PATH = Path(__file__).parent / "data" / "indexes" / "faiss"


def build_vector_store(
    chunks: list[Document] | None = None,
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> FAISS:
    """Embed chunks, build a FAISS index, and persist it locally."""
    documents = chunks if chunks is not None else chunk_documents(load_paper())
    if not documents:
        raise ValueError("Cannot build a vector store without documents")

    vector_store = FAISS.from_documents(documents, create_embedder())

    destination = Path(index_path)
    destination.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(destination))
    return vector_store


@lru_cache(maxsize=4)
def load_vector_store(index_path: str | Path = DEFAULT_INDEX_PATH) -> FAISS:
    """Load a locally generated FAISS index."""
    return FAISS.load_local(
        str(index_path),
        create_embedder(),
        allow_dangerous_deserialization=True,
    )


def search(
    query: str,
    k: int = 4,
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> list[tuple[Document, float]]:
    """Return the closest chunks and their FAISS distance scores."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")

    return load_vector_store(index_path).similarity_search_with_score(query, k=k)


if __name__ == "__main__":
    store = build_vector_store()
    print(f"Stored {store.index.ntotal} vectors in {DEFAULT_INDEX_PATH}")

    results = search("What accuracy did VGG16 achieve?")
    print(f"Top result score: {results[0][1]:.4f}")
    print(results[0][0].page_content[:300])
