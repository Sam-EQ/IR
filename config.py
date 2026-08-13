import os
from huggingface_hub import login as _hf_login

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    _hf_login(token=HF_TOKEN, add_to_git_credential=False)

OPENSEARCH_HOST = os.environ["OPENSEARCH_HOST"]
OPENSEARCH_USER = os.environ["OPENSEARCH_USER"]
OPENSEARCH_PASS = os.environ["OPENSEARCH_PASS"]
OPENSEARCH_INDEX = os.environ.get("OPENSEARCH_INDEX", "legal_toolkit")
TEXT_FIELD = os.environ.get("OPENSEARCH_TEXT_FIELD", "text")

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_RERANK_MODEL = "gemini-3.6-flash"

PROMPTRIEVER_MODEL = "samaya-ai/promptriever-llama3.1-8b-v1"
COLBERT_MODEL = "lightonai/GTE-ModernColBERT-v1"

BM25_TOP_N = 50
DENSE_TOP_N = 50
RERANK_TOP_K = 10

PROMPTRIEVER_BATCH_SIZE = int(os.environ.get("PROMPTRIEVER_BATCH_SIZE", "4"))

INDEX_DIR = os.environ.get("DENSE_INDEX_DIR", "./dense_indexes")