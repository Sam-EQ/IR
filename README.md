# Flow A vs Flow B — retrieval comparison

Tests these two flows against the same OpenSearch corpus and shows the
real-time difference in what each one retrieves:

```
Flow A: BM25 + Promptriever Llama3 8B (4096-d)  -> union -> Gemini 2.5 Pro -> top-k
Flow B: BM25 + GTE-ModernColBERT                -> union -> Gemini 2.5 Pro -> top-k
```

No generator step — this is retrieval-only, per what was asked.

## Setup

Run on the EC2 L4 GPU box. Neither Promptriever nor GTE-ModernColBERT is
available through any HuggingFace Inference Provider (checked both model
pages — neither lists one), so this always means loading the actual weights
and running them locally. There's no shortcut here.

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OpenSearch creds, GEMINI_API_KEY, HF_TOKEN
export $(cat .env | xargs)
```

**Creds you need:**
- `GEMINI_API_KEY` — both flows specify Gemini 2.5 Pro as the reranker.
- `HF_TOKEN` — Promptriever's base model (`meta-llama/Meta-Llama-3.1-8B`) is
  **gated**. You need to (1) accept the license at
  huggingface.co/meta-llama/Meta-Llama-3.1-8B with the account tied to your
  token, then (2) put that token in `.env`. GTE-ModernColBERT isn't gated,
  but the same token works fine for both.

**Sizing note:** Promptriever is 8B params (~16GB in bf16 just for weights).
On a 24GB L4 that leaves limited headroom — `PROMPTRIEVER_BATCH_SIZE`
defaults to 4 in `.env.example`. If `nvidia-smi` shows you have room, bump it
up for faster indexing; if you hit a CUDA OOM, drop it to 1–2.

## Run order

1. **Build the indexes once** (pulls the whole corpus from OpenSearch, encodes
   it with each dense model, caches to disk):

   ```bash
   python build_dense_index.py --model both
   ```

   This is the slow step (especially Promptriever — 8B params). Only needs
   rerunning if the corpus changes.

2. **Compare the two flows on a query:**

   ```bash
   python compare.py "who signed the indemnification clause"
   ```

   Or batch a list of test queries:

   ```bash
   python compare.py --file queries.txt
   ```

   Output shows, per flow: BM25/dense/union candidate counts, the reranked
   top-k with snippets, then an overlap/diff summary (which doc IDs only
   Flow A found, which only Flow B found).

## File map

| File | What it does |
|---|---|
| `config.py` | All env vars / model names / params in one place |
| `opensearch_utils.py` | BM25 search + full-corpus fetch (shared by both flows) |
| `doc_store.py` | Cached id -> text map, so the reranker sees full text |
| `dense_promptriever.py` | Bi-encoder dense retrieval (Flow A) |
| `dense_colbert.py` | Multi-vector / MaxSim dense retrieval via `pylate` (Flow B) |
| `reranker.py` | Gemini 2.5 Pro rerank, shared by both flows |
| `pipeline.py` | Wires BM25 + one dense backend + rerank into a flow |
| `build_dense_index.py` | One-time corpus encode/index step |
| `compare.py` | Runs both flows on a query, prints side-by-side + diff |

## Assumptions made (flag if wrong)

- OpenSearch index defaults to `legal_toolkit` with text field `content` —
  override via `OPENSEARCH_INDEX` / `OPENSEARCH_TEXT_FIELD` in `.env` if
  Pradhap's eval corpus lives elsewhere.
- BM25/dense candidate pools are top-50 each before union; final rerank
  output is top-10. Both are just constants in `config.py`.
