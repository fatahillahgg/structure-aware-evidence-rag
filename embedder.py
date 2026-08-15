from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from chunker import chunk_documents
from loader import load_paper


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embedder() -> HuggingFaceEmbeddings:
    """Create the local, open-source embedding model."""
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_documents(
    documents: list[Document],
    embedder: HuggingFaceEmbeddings | None = None,
) -> list[list[float]]:
    """Embed document content while keeping metadata on the original documents."""
    model = embedder or create_embedder()
    return model.embed_documents([document.page_content for document in documents])


if __name__ == "__main__":
    chunks = chunk_documents(load_paper())
    vectors = embed_documents(chunks)
    print(f"Embedded {len(vectors)} chunks")
    print(f"Embedding dimensions: {len(vectors[0])}")
