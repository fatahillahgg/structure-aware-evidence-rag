import argparse
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from retrieval import DEFAULT_INDEX_PATH, retrieve


SYSTEM_PROMPT = """You answer questions about the supplied research paper.
Use only the context provided below. If the answer is not in the context, say
that you do not have enough information in the paper. Do not invent facts.
Keep the answer concise and mention relevant numbers when available."""


@lru_cache(maxsize=1)
def create_llm() -> ChatOpenAI:
    """Create a Gemini chat model through OpenRouter."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_api_key_here":
        raise RuntimeError("Set OPENROUTER_API_KEY in .env before running the RAG")

    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash"),
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_SITE_NAME", "rag-paper"),
        },
    )


def answer_question(
    question: str,
    k: int = 4,
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> str:
    """Retrieve paper context and generate a grounded answer."""
    results = retrieve(question, k=k, index_path=index_path)
    context = "\n\n---\n\n".join(
        f"Source chunk {number}:\n{document.page_content}"
        for number, (document, _) in enumerate(results, start=1)
    )

    response = create_llm().invoke(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                f"Context from the paper:\n\n{context}\n\nQuestion: {question}",
            ),
        ]
    )
    return str(response.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions using naive RAG")
    parser.add_argument("question", help="Question about the research paper")
    parser.add_argument("-k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    args = parser.parse_args()

    print(answer_question(args.question, k=args.k, index_path=args.index_path))
