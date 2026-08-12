"""
All configuration in one place. Everything comes from env vars so nothing
sensitive lives in the code. Copy .env.example to .env and fill it in.
"""
import os
from huggingface_hub import login as _hf_login

# ---- HuggingFace (Promptriever's base model, meta-llama/Meta-Llama-3.1-8B,
#      is gated — you need a token AND to have accepted the license on the
#      model page at huggingface.co/meta-llama/Meta-Llama-3.1-8B) ----
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    _hf_login(token=HF_TOKEN, add_to_git_credential=False)

# ---- OpenSearch ----
OPENSEARCH_HOST = os.environ["OPENSEARCH_HOST"]          # e.g. "https://your-domain-endpoint:9200"
OPENSEARCH_USER = os.environ["OPENSEARCH_USER"]
OPENSEARCH_PASS = os.environ["OPENSEARCH_PASS"]
OPENSEARCH_INDEX = os.environ.get("OPENSEARCH_INDEX", "legal_toolkit")
TEXT_FIELD = os.environ.get("OPENSEARCH_TEXT_FIELD", "content")

# ---- Gemini (both flows use Gemini 2.5 Pro as the reranker — required) ----
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_RERANK_MODEL = "gemini-2.5-pro"

# ---- Dense models (exactly the ones named in the two flows) ----
PROMPTRIEVER_MODEL = "samaya-ai/promptriever-llama3.1-8b-v1"
COLBERT_MODEL = "lightonai/GTE-ModernColBERT-v1"

# ---- Retrieval params ----
BM25_TOP_N = 50
DENSE_TOP_N = 50
RERANK_TOP_K = 10

# ---- GPU-sensitive params ----
# Promptriever is 8B params in bf16 (~16GB just for weights). On a 24GB L4
# this leaves limited headroom for batched encoding — start small and raise
# it if nvidia-smi shows you have room to spare.
PROMPTRIEVER_BATCH_SIZE = int(os.environ.get("PROMPTRIEVER_BATCH_SIZE", "4"))

# ---- Where precomputed dense indexes / doc store get cached ----
INDEX_DIR = os.environ.get("DENSE_INDEX_DIR", "./dense_indexes")
