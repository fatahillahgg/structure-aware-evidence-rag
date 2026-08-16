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

Citra was used to extract the PDF into `data/processed/cleaned_paper.md`, removing PDF layout noise before LangChain reads it. The original PDF is kept in `data/raw/`.

```bash
uv run python loader.py
```

The loader returns the full paper as a LangChain `Document` and converts the YAML front matter into document metadata.

## Chunk the Paper

```bash
uv run python chunker.py
```

The chunker first follows the Markdown heading hierarchy, then uses `RecursiveCharacterTextSplitter` within each section with 1,000-character chunks and 200-character overlap. Each chunk keeps document metadata plus `chunk_id`, section path, section level, content type, and character offsets.

## Create Embeddings

```bash
uv run python embedder.py
```

This uses the open-source `sentence-transformers/all-MiniLM-L6-v2` model locally. The model is downloaded on its first run, and vectors are normalized for cosine similarity.

## Build the FAISS Store

```bash
uv run python vector_store.py
```

This persists the chunk embeddings and metadata under `data/indexes/faiss/`. The generated index is ignored by git and can be rebuilt from the cleaned Markdown source.

## Retrieve Relevant Chunks

Build the index first, then run a semantic search:

```bash
uv run python retrieval.py "What accuracy did VGG16 achieve?" -k 4
```

Use `retrieve()` when scores and document metadata are needed, or `build_context()` to produce prompt-ready context for a RAG pipeline.

## Ask Questions With Naive RAG

The RAG flow first analyzes the user query. Standalone questions use direct retrieval, context-dependent questions use history-aware rewriting, and multi-part questions are decomposed. The selected queries are sent to dense FAISS search and sparse BM25 search, combined with Reciprocal Rank Fusion (RRF), reranked with a local CrossEncoder, normalized and deduplicated, evaluated for context sufficiency, and sent as cited context to Gemini through OpenRouter:

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

Run the full evaluation against `data/evaluation/dataset.json`:

```bash
uv run python evaluate.py
```

This generates results under `data/evaluation/latest/`. To evaluate retrieval without answer-generation calls (rewrite-routed questions may still call OpenRouter):

```bash
uv run python evaluate.py --retrieval-only
```

Compare direct retrieval with query rewriting:

```bash
uv run python evaluate.py --retrieval-only --query-mode direct --output-dir data/evaluation/runs/direct
uv run python evaluate.py --retrieval-only --query-mode rewrite --output-dir data/evaluation/runs/rewrite
uv run python evaluate.py --retrieval-only --query-mode analyzer --output-dir data/evaluation/runs/analyzer
```

The retriever combines dense FAISS similarity and sparse BM25 rankings with RRF, then applies `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank the candidate chunks. The evaluation command prints the current scores for the 30-question dataset; generation scores can vary slightly between model responses.

Baseline retrieval after evidence deduplication:

- Direct query: **72.4%** retrieval keyword recall
- Rewritten query: **74.1%** retrieval keyword recall

The context sufficiency evaluator runs after normalized evidence selection and records `sufficient`, `partial`, `insufficient`, or `conflicting` assessments in the trace. The retrieval controller uses these assessments to trigger bounded corrective retrieval or abstention.

The retrieval controller consumes that assessment and chooses one bounded action: `ANSWER`, `REWRITE`, `EXPAND`, `DECOMPOSE`, `RETRY_DIRECT`, or `ABSTAIN`. Corrective retrieval is limited to two attempts per request.

Run the final answer-quality evaluation through the controller:

```bash
uv run python evaluate.py --query-mode analyzer --output-dir data/evaluation/runs/final-controller-calibrated
```

## Observability

Enable structured JSONL tracing by setting `RAG_TRACE_PATH` in `.env`:

```env
RAG_TRACE_PATH=data/traces/rag.jsonl
```

Each request records a `trace_id`, query route, rewrite/decomposition timing, dense and BM25 retrieval timing, reranking, fallback usage, selected context, answer generation timing, and request duration. Tracing is disabled when `RAG_TRACE_PATH` is empty or unset.

Analyzer routing accuracy and results are stored under `data/evaluation/runs/analyzer/`.

Run the network-free unit tests:

```bash
uv run python -m unittest discover -s tests -v
```
