# RAG Project Report

## Summary

Implemented an end-to-end retrieval-augmented generation system for the lung cancer CT scan research paper.

The final retrieval pipeline is hybrid search with reranking:

```text
Dense FAISS search + Sparse BM25 search + Reciprocal Rank Fusion (RRF) + CrossEncoder reranking
```

Retrieved context is sent to Gemini through OpenRouter to generate grounded answers.

## Work Completed

- Loaded the cleaned research paper from `data/cleaned_paper.md`.
- Split the paper into overlapping chunks.
- Generated local embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Built and saved a FAISS index under `data/faiss_index/`.
- Added dense semantic retrieval with FAISS.
- Added sparse lexical retrieval with Okapi BM25.
- Combined dense and sparse rankings with RRF.
- Reranked the RRF candidate pool with a local CrossEncoder.
- Added Gemini access through OpenRouter.
- Added a CLI for question answering in `rag.py`.
- Added a Gradio chatbot in `app.py`.
- Added evaluation using `data/eval.json`.
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
│   ├── cleaned_paper.md   # Source paper
│   ├── eval.json          # Evaluation questions and expected answers
│   ├── faiss_index/       # Generated FAISS index
│   └── evaluation/        # Evaluation chart and results
└── .env.example           # Environment variable template
```

## Retrieval Architecture

1. The query is embedded with the same MiniLM model used for document chunks.
2. FAISS returns dense semantic matches.
3. BM25 returns lexical matches based on terms in the query.
4. RRF merges both ranked lists using:

   ```text
   RRF score = 1 / (60 + rank)
   ```

5. The fused candidates are reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2` using query-chunk relevance scores.
6. The highest-ranked reranked chunks are passed to the language model.

## Evaluation

The evaluation dataset contains 24 questions across direct retrieval, methodology, results, comparison, limitations, and multi-hop categories.

Latest full evaluation:

- Retrieval keyword recall: **77.4%**
- Answer keyword recall: **73.8%**
- Answer token F1: **42.2%**

The CrossEncoder reranker improved retrieval keyword recall from the previous RRF score of **68.9%** to **77.4%**. Compared with the previous full evaluation, answer keyword recall increased from **57.5%** to **73.8%**, while answer token F1 increased from **41.5%** to **42.2%**.

Generated files:

- `data/evaluation/naive_rag_performance.png`
- `data/evaluation/results.json`

The answer metrics are lexical metrics, so they do not fully measure semantic correctness. Generation scores can also vary slightly between model responses.

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

Evaluate retrieval without OpenRouter calls:

```bash
uv run python evaluate.py --retrieval-only
```

## Current Limitations

- The RRF constant, candidate depth, and reranker model are fixed defaults.
- BM25 is rebuilt in memory when the process starts.
- Evaluation uses keyword recall and token F1 rather than a semantic judge.
- The chatbot answers each turn independently and does not use conversation history.
- The system currently supports one indexed research paper.
