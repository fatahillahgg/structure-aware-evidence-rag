import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loader import load_paper


_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_FIGURE_PATTERN = re.compile(r"\bfigure\s+\d+\b", re.IGNORECASE)
_TABLE_PATTERN = re.compile(r"\btable\s+\d+\b", re.IGNORECASE)
_EQUATION_PATTERN = re.compile(r"(?:=|\b(?:loss|accuracy|precision|recall)\b)", re.IGNORECASE)


@dataclass(frozen=True)
class _Section:
    heading: str
    level: int
    path: tuple[str, ...]
    text: str
    start: int


def _find_sections(content: str) -> list[_Section]:
    headings = list(_HEADING_PATTERN.finditer(content))
    if not headings:
        return [_Section("Document", 0, ("Document",), content, 0)]

    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    for index, match in enumerate(headings):
        level = len(match.group(1))
        heading = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        end = headings[index + 1].start() if index + 1 < len(headings) else len(content)
        sections.append(
            _Section(
                heading=heading,
                level=level,
                path=tuple(item[1] for item in stack),
                text=content[match.start() : end].strip(),
                start=match.start(),
            )
        )
    return sections


def _content_type(text: str) -> str:
    if _TABLE_PATTERN.search(text):
        return "table"
    if _FIGURE_PATTERN.search(text):
        return "figure"
    return "text"


def _base_metadata(document: Document) -> dict:
    metadata = dict(document.metadata)
    metadata.setdefault("document_type", "research_paper")
    metadata.setdefault("source_type", "cleaned_markdown")
    return metadata


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split a paper by headings while preserving rich structural metadata."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks: list[Document] = []
    for document_index, document in enumerate(documents):
        base_metadata = _base_metadata(document)
        sections = _find_sections(document.page_content)
        for section_index, section in enumerate(sections):
            section_document = Document(
                page_content=section.text,
                metadata={
                    **base_metadata,
                    "section": section.heading,
                    "section_path": list(section.path),
                    "section_level": section.level,
                    "section_index": section_index,
                    "content_type": _content_type(section.text),
                },
            )
            section_chunks = splitter.split_documents([section_document])
            for chunk_index, chunk in enumerate(section_chunks):
                local_start = int(chunk.metadata.get("start_index", 0))
                chunk.metadata.update(
                    {
                        "chunk_id": f"document-{document_index}-chunk-{len(chunks)}",
                        "chunk_index": len(chunks),
                        "section_chunk_index": chunk_index,
                        "start_index": section.start + local_start,
                        "end_index": section.start + local_start + len(chunk.page_content),
                        "has_equation_or_metric": bool(_EQUATION_PATTERN.search(chunk.page_content)),
                    }
                )
                chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    chunks = chunk_documents(load_paper())
    print(f"Created {len(chunks)} chunks")
    print(f"First chunk metadata: {chunks[0].metadata}")
    print(f"First chunk characters: {len(chunks[0].page_content)}")
