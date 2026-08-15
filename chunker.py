from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loader import load_paper


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents while preserving metadata and recording start offsets."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    chunks = chunk_documents(load_paper())
    print(f"Created {len(chunks)} chunks")
    print(f"First chunk metadata: {chunks[0].metadata}")
    print(f"First chunk characters: {len(chunks[0].page_content)}")
