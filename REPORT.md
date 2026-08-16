# RAG Project Report

## Summary

Implemented an end-to-end retrieval-augmented generation system for the lung cancer CT scan research paper.

The final RAG pipeline is query-analyzed hybrid search with reranking:

```text
Query Analyzer (Direct or Rewrite) + Dense FAISS search + Sparse BM25 search + Reciprocal Rank Fusion (RRF) + CrossEncoder reranking
```

Retrieved context is sent to Gemini through OpenRouter to generate grounded answers.

## Work Completed

- Loaded the cleaned research paper from `data/processed/cleaned_paper.md`.
- Split the paper by Markdown section hierarchy, then into overlapping chunks.
- Preserved front matter and structural metadata including section path, chunk ID, content type, metrics, and character offsets.
- Generated local embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Built and saved a FAISS index under `data/indexes/faiss/`.
- Added dense semantic retrieval with FAISS.
- Added sparse lexical retrieval with Okapi BM25.
- Combined dense and sparse rankings with RRF.
- Reranked the RRF candidate pool with a local CrossEncoder.
- Added query analysis to route standalone questions directly and rewrite context-dependent questions.
- Added LLM-based query rewriting for the rewrite route.
- Added history-aware rewriting for follow-up questions.
- Added multi-part query decomposition and result merging.
- Added low-score retrieval fallback from direct search to rewriting.
- Added evidence normalization and exact/near-duplicate removal after reranking.
- Added context sufficiency evaluation with sufficient, partial, insufficient, and conflicting statuses.
- Added source chunk citations to generated answers.
- Added semantic answer similarity evaluation using the local embedding model.
- Added optional structured JSONL observability with per-request trace IDs and stage timings.
- Added Gemini access through OpenRouter.
- Added a CLI for question answering in `rag.py`.
- Added a Gradio chatbot in `app.py`.
- Added evaluation using `data/evaluation/dataset.json`.
- Added labeled query-routing evaluation for the 30-question dataset.
- Generated a performance chart and raw evaluation results.
- Added `.env.example` configuration.

## Project Structure

```text
rag_paper/
├── app.py                 # Gradio chatbot
├── rag.py                 # RAG generation with Gemini/OpenRouter
├── retrieval.py           # Dense, BM25, and RRF hybrid retrieval
├── vector_store.py        # FAISS index creation and loading
├── embedder.py            # Local sentence embeddings
├── chunker.py             # Document chunking
├── loader.py              # Markdown document loading
├── evaluate.py            # Evaluation metrics and chart generation
├── data/
│   ├── raw/               # Original source files
│   ├── processed/         # Cleaned paper used by the pipeline
│   ├── indexes/           # Generated retrieval indexes
│   └── evaluation/        # Dataset and evaluation runs
└── .env.example           # Environment variable template
```

## Retrieval Architecture

1. The query analyzer chooses direct retrieval, history-aware rewriting, or multi-part decomposition.
2. Direct retrieval can fall back to rewriting when the top reranker score is low.
3. Reranked candidates are normalized and deduplicated before context evaluation.
4. The context evaluator reports supported, missing, conflicting, and relevant evidence IDs.
5. The selected retrieval query is embedded with the same MiniLM model used for document chunks.
6. FAISS returns dense semantic matches.
7. BM25 returns lexical matches based on terms in the selected query.
8. RRF merges both ranked lists using:

   ```text
   RRF score = 1 / (60 + rank)
   ```

9. The fused candidates are reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2` using query-chunk relevance scores.
10. The highest-ranked reranked chunks are passed to the language model with source chunk labels.

## Evaluation

The evaluation dataset contains 30 questions across direct retrieval, methodology, results, comparison, limitations, multi-hop, ambiguous follow-up, and multi-part categories.

Latest full evaluation:

- Retrieval keyword recall: **77.0%**
- Answer keyword recall: **71.3%**
- Answer token F1: **28.0%**
- Answer semantic similarity: **69.1%**
- Answers with citations: **100.0%**

Query analyzer evaluation:

- Routing accuracy: **100.0%** on the labeled 30-question dataset.
- Analyzer retrieval keyword recall: **77.0%** after structure-aware reindexing.
- Analyzer answer keyword recall: **71.3%**.
- Analyzer answer semantic similarity: **69.1%**.
- Answers with citations: **100.0%**.

The structure-aware evaluation uses a rebuilt 197-vector FAISS index with section metadata. Scores can differ from the earlier baseline because the expanded dataset and structure-aware index change the retrieval context.

Generated files:

- `data/evaluation/latest/naive_rag_performance.png`
- `data/evaluation/latest/results.json`

The latest structure-aware full evaluation is stored under `data/evaluation/runs/structure-aware-analyzer-full/`.

Baseline retrieval after evidence deduplication:

- Direct query: **72.4%** retrieval keyword recall.
- Rewritten query: **74.1%** retrieval keyword recall.

The answer metrics are lexical and embedding-based approximations, so they do not replace human or LLM judging. Generation scores can also vary slightly between model responses.

## Configuration

Copy the template and add an OpenRouter key:

```bash
cp .env.example .env
```

Important variables:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=google/gemini-2.5-flash
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## How To Run

Install dependencies:

```bash
uv sync
```

Rebuild the FAISS index:

```bash
uv run python vector_store.py
```

Run retrieval:

```bash
uv run python retrieval.py "What accuracy did VGG16 achieve?"
```

Ask a question through the CLI RAG:

```bash
uv run python rag.py "What accuracy did VGG16 achieve?"
```

Start the Gradio chatbot:

```bash
uv run python app.py
```

Evaluate the system:

```bash
uv run python evaluate.py
```

Evaluate retrieval without answer-generation calls (rewrite-routed questions may still call OpenRouter):

```bash
uv run python evaluate.py --retrieval-only
```

## Current Limitations

- Context-dependent and multi-part questions routed to rewriting/decomposition require an LLM call before retrieval.
- The RRF constant, candidate depth, and reranker model are fixed defaults.
- BM25 is rebuilt in memory when the process starts.
- The analyzer is rule-based and may need more labeled examples as query language expands.
- The fallback threshold is currently a fixed default and needs tuning on a larger validation set.
- Tracing is opt-in through `RAG_TRACE_PATH` and currently writes local JSONL rather than exporting to an external tracing backend.
- The system currently supports one indexed research paper.
