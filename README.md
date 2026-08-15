# RAG Paper

Python environment managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync
source .venv/bin/activate
cp .env.example .env
```

Add your OpenRouter API key to `.env`. Gemini will be accessed through OpenRouter using the configured `OPENROUTER_MODEL`.
If Hugging Face requires authentication when downloading the reranker, also add `HF_TOKEN` to `.env`.

## Load the Cleaned Paper

Citra was used to extract the PDF into `data/cleaned_paper.md`, removing PDF layout noise before LangChain reads it.

```bash
uv run python loader.py
```

The loader returns the full paper as a LangChain `Document` and converts the YAML front matter into document metadata.

## Chunk the Paper

```bash
uv run python chunker.py
```

The chunker uses `RecursiveCharacterTextSplitter` with 1,000-character chunks and 200-character overlap.

## Create Embeddings

```bash
uv run python embedder.py
```

This uses the open-source `sentence-transformers/all-MiniLM-L6-v2` model locally. The model is downloaded on its first run, and vectors are normalized for cosine similarity.

## Build the FAISS Store

```bash
uv run python vector_store.py
```

This persists the chunk embeddings and metadata under `data/faiss_index/`. The generated index is ignored by git and can be rebuilt from the cleaned Markdown source.

## Retrieve Relevant Chunks

Build the index first, then run a semantic search:

```bash
uv run python retrieval.py "What accuracy did VGG16 achieve?" -k 4
```

Use `retrieve()` when scores and document metadata are needed, or `build_context()` to produce prompt-ready context for a RAG pipeline.

## Ask Questions With Naive RAG

The RAG flow first rewrites the user query for retrieval, then combines dense FAISS search and sparse BM25 search with Reciprocal Rank Fusion (RRF), reranks the candidates with a local CrossEncoder, and sends the top chunks as context to Gemini through OpenRouter:

```bash
uv run python rag.py "What accuracy did VGG16 achieve?"
```

## Gradio Chatbot

Start the web chatbot:

```bash
uv run python app.py
```

Open `http://127.0.0.1:7860` in your browser. The app uses the same FAISS index and OpenRouter Gemini RAG pipeline as the CLI.

## Evaluate Performance

Run the full evaluation against `data/eval.json`:

```bash
uv run python evaluate.py
```

This generates `data/evaluation/naive_rag_performance.png` and `data/evaluation/results.json`. To evaluate retrieval without making OpenRouter calls:

```bash
uv run python evaluate.py --retrieval-only
```

The retriever combines dense FAISS similarity and sparse BM25 rankings with RRF, then applies `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank the candidate chunks. The evaluation command prints the current scores for the 24-question dataset; generation scores can vary slightly between model responses.

Run the network-free unit tests:

```bash
uv run python -m unittest discover -s tests -v
```
