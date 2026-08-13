# Flow A vs Flow B — retrieval comparison

Tests these two flows against the same OpenSearch corpus and shows the real-time difference in what each one retrieves:

```
Flow A: BM25 + Promptriever Llama3 8B (4096-d)  -> union -> Gemini rerank -> top-k
Flow B: BM25 + GTE-ModernColBERT                -> union -> Gemini rerank -> top-k

```

No generator step — this is retrieval-only, per what was asked. The goal is to see, per query, exactly which documents each flow retrieves and where they diverge — not to generate an answer.

**Note on the reranker model:** the flows were originally specified with Gemini 2.5 Pro. That model is blocked for newer API projects and is being fully retired by Google on Oct 16, 2026, so both flows now use **Gemini 3.1 Pro** (`gemini-3.1-pro-preview`) — the current flagship Pro-tier model, same role, closest live equivalent. Note Pro-tier models have **no free quota**— billing must be enabled on the project behind `GEMINI_API_KEY`, or swap `GEMINI_RERANK_MODEL` in `config.py` to `gemini-3.6-flash`, which works on the free tier.

## Setup

Run on the EC2 L4 GPU box. Neither Promptriever nor GTE-ModernColBERT is available through any HuggingFace Inference Provider (checked both model pages — neither lists one), so this always means loading the actual weights and running them locally. There's no shortcut here.

```bash
python3.11 -m venv doc
source doc/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OpenSearch creds, GEMINI_API_KEY, HF_TOKEN

```

**Creds you need:**

- `GEMINI_API_KEY` — both flows use Gemini as the reranker (see note above on model/billing).
- `HF_TOKEN` — Promptriever's base model (`meta-llama/Llama-3.1-8B`) is **gated**. You need to (1) accept the license at huggingface.co/meta-llama/Llama-3.1-8B with the account tied to your token, then (2) put that token in `.env`. GTE-ModernColBERT isn't gated, but the same token works fine for both.

**Sizing note:** Promptriever is 8B params (~16GB in bf16 just for weights). On a box with limited system RAM (not just VRAM), loading it naively can OOM-kill — see Troubleshooting below for why the code avoids `merge_and_unload()`. `PROMPTRIEVER_BATCH_SIZE` in `.env.example` defaults to 4; bump it up if `nvidia-smi` shows spare VRAM, drop it to 1-2 if you hit a CUDA OOM.

## Making .env load automatically

Env vars exported in one terminal session don't carry over when you open a new shell or re-activate the venv later — this trips people up constantly. Fix it once by appending auto-sourcing to the venv's own activate script:

```bash
set -a
source /home/ubuntu/IR/.env
set +a
```

After this, every `source doc/bin/activate` also loads `.env` automatically — no more `KeyError: 'OPENSEARCH_HOST'` from a fresh shell. Verify it worked:

```bash
source doc/bin/activate
echo $OPENSEARCH_HOST
```



## Run order

1. **Build the indexes once** (pulls the whole corpus from OpenSearch, encodes it with each dense model, caches to disk):
  ```bash
  python build_dense_index.py --model both
  ```
  This is the slow step (especially Promptriever — 8B params, expect it to take a while with no progress bar during encoding). **You only rerun this if the OpenSearch corpus itself changes** — everyday comparisons just reload the cached index, no re-encoding. You can build one model at a time with `--model promptriever` or `--model colbert` if one is blocked (e.g. waiting on gated-repo approval) while the other isn't.
2. **Compare the two flows on a query — this is the actual deliverable:**
  ```bash
  python compare.py "who signed the indemnification clause"
  ```
  Or batch a list of real queries:
  ```bash
  python compare.py --file legal_toolkit_queries.txt
  ```
  Output shows, per flow: BM25/dense/union candidate counts, the reranked top-k with snippets, then an overlap/diff summary — which doc IDs both flows agreed on, which only Flow A found, which only Flow B found. That diff is the real-time retrieval comparison this whole project exists to produce.
  Note: Promptriever reloads into GPU memory fresh each time you start `compare.py` (~1.5 min) — the cached *index*isn't rebuilt, just the model itself gets reloaded into memory each run.



## What's actually stored on disk after indexing

- `promptriever_embeddings.npy` — a `(35915, 4096)` matrix: one row per document, each row a 4096-number vector (its position in Promptriever's meaning-space). `promptriever_ids.npy` — the matching OpenSearch doc ID for each row, same order. Together: precomputed coordinates for every doc, so query time is just "embed the query, dot-product against all 35,915 rows" instead of re-running an 8B model per query.
- `colbert_index/` — a Voyager-managed folder holding a vector *per token* per document (not one pooled vector per doc), plus a nearest-neighbor search structure over all of them, so MaxSim lookups at query time are fast instead of brute-force.



## File map


| File                        | What it does                                                                                              |
| --------------------------- | --------------------------------------------------------------------------------------------------------- |
| `config.py`                 | All env vars / model names / params in one place                                                          |
| `opensearch_utils.py`       | BM25 search + full-corpus fetch (shared by both flows)                                                    |
| `doc_store.py`              | Cached id -> text map, so the reranker sees full text                                                     |
| `dense_promptriever.py`     | Bi-encoder dense retrieval (Flow A) — manual `transformers`+`peft` load, not `mteb` (see Troubleshooting) |
| `dense_colbert.py`          | Multi-vector / MaxSim dense retrieval via `pylate` (Flow B)                                               |
| `reranker.py`               | Gemini rerank shared by both flows, with retry-on-transient-error                                         |
| `pipeline.py`               | Wires BM25 + one dense backend + rerank into a flow                                                       |
| `build_dense_index.py`      | One-time corpus encode/index step                                                                         |
| `compare.py`                | Runs both flows on a query, prints side-by-side + diff                                                    |
| `legal_toolkit_queries.txt` | Real user queries pulled from actual Legal Toolkit conversation logs                                      |




## Troubleshooting (things that actually happened building this)

- `KeyError: 'OPENSEARCH_HOST'` — env vars not loaded in the current shell. Either you skipped `source .env` this session, or (more likely if you did source it before) you opened a new terminal/re-activated the venv, which resets exported vars. See "Making .env load automatically" above.
- `OPENSEARCH_HOST` **has a trailing slash** — the OpenSearch client is picky about this. Strip it: `sed -i 's#/$##' .env` then re-source.
- `mteb.get_model()` **throws** `takes 0 positional arguments but 1 was given` — an internal API break in the installed `mteb`version's model loader. `dense_promptriever.py` bypasses `mteb` entirely and loads Promptriever directly via `transformers` + `peft`, mirroring the manual loading code on the model's own HF card. More robust long-term since it doesn't depend on `mteb`'s internal loader signature.
- `GatedRepoError: 403` **on** `meta-llama/Llama-3.1-8B` — you need to (1) accept Meta's license at huggingface.co/meta-llama/Llama-3.1-8B logged in as the account behind your `HF_TOKEN`, and (2) wait for approval (usually minutes). Verify access before rerunning anything:
  ```bash
  python -c "from huggingface_hub import HfApi; print(HfApi().model_info('meta-llama/Llama-3.1-8B'))"

  ```
- **Process silently** `Killed` **with no traceback while loading Promptriever** — this is the Linux OOM killer hitting *system RAM* (not VRAM). `merge_and_unload()` briefly needs ~2x the model's memory on CPU before moving to GPU, which is enough to OOM on boxes with limited system RAM. Fixed by loading with `low_cpu_mem_usage=True` + `device_map={"": 0}` and skipping the merge step entirely (LoRA applied at forward time instead — functionally identical output, no memory spike). Check with `free -h` / `nvidia-smi` if this recurs.
- `ModuleNotFoundError: No module named 'voyager'` — ColBERT's index backend is an optional extra. `pip install "pylate[voyager]"`.
- `404 NotFound: models/gemini-2.5-pro is no longer available to new users` — Google blocked `gemini-2.5-pro` for newer API projects/accounts ahead of its Oct 16, 2026 full retirement, and deprecated the `google.generativeai` SDK in favor of `google-genai`. Fixed by switching to `gemini-3.1-pro-preview` on the new SDK.
- `429 RESOURCE_EXHAUSTED ... limit: 0 ... free_tier` — Pro-tier Gemini models have zero free-tier quota, period. Either enable billing on the project behind `GEMINI_API_KEY`, or set `GEMINI_RERANK_MODEL` in `config.py` to `gemini-3.6-flash`, which does run on the free tier.
- `503 UNAVAILABLE: high demand` — transient Gemini server-side hiccup, not something in our control. `reranker.py` now retries automatically with exponential backoff (4 attempts) so a single flaky call doesn't kill a multi-query batch run.
- `No Promptriever index found` **/** `No doc store found` — you ran `compare.py` before `build_dense_index.py` finished successfully for that model. Rerun the index build.

