from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
import yaml


DEFAULT_DOCUMENT = Path(__file__).parent / "data" / "processed" / "cleaned_paper.md"


def load_paper(path: str | Path = DEFAULT_DOCUMENT) -> list[Document]:
    """Load the Citra-cleaned paper and attach its front matter as metadata."""
    document_path = Path(path)
    if not document_path.is_file():
        raise FileNotFoundError(f"Cleaned document not found: {document_path}")

    documents = TextLoader(str(document_path), encoding="utf-8").load()
    metadata, content = _read_front_matter(documents[0].page_content)
    documents[0].page_content = content
    documents[0].metadata.update(metadata)
    return documents


def _read_front_matter(content: str) -> tuple[dict, str]:
    """Extract YAML front matter without leaving it in the document content."""
    if not content.startswith("---\n"):
        return {}, content

    _, front_matter, body = content.split("---\n", 2)
    metadata = yaml.safe_load(front_matter) or {}
    if not isinstance(metadata, dict):
        raise ValueError("Document front matter must be a YAML mapping")
    return metadata, body.lstrip()


if __name__ == "__main__":
    documents = load_paper()
    print(f"Loaded {len(documents)} document(s) from {documents[0].metadata['source']}")
    print(f"Characters: {len(documents[0].page_content)}")
