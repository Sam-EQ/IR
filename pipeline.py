"""
Wires BM25 + one dense retriever + Gemini rerank into a single callable
pipeline, matching the two flows exactly:

  Flow A: BM25 + Promptriever Llama3 8B  -> union -> Gemini 2.5 Pro -> top-k
  Flow B: BM25 + GTE-ModernColBERT       -> union -> Gemini 2.5 Pro -> top-k
"""
import opensearch_utils
import dense_promptriever
import dense_colbert
import reranker
import doc_store

DENSE_BACKENDS = {
    "promptriever": dense_promptriever,
    "colbert": dense_colbert,
}

_doc_text_cache = None


def _get_doc_text_map():
    global _doc_text_cache
    if _doc_text_cache is None:
        _doc_text_cache = doc_store.load()
    return _doc_text_cache


def _union(bm25_hits, dense_hits):
    """Dedup by doc id, keeping the higher score if a doc shows up in both."""
    merged = {}
    for h in bm25_hits + dense_hits:
        if h["id"] not in merged or h["score"] > merged[h["id"]]["score"]:
            merged[h["id"]] = h
    return list(merged.values())


def run_pipeline(query_text, dense_backend, os_client, top_k=None):
    """dense_backend: "promptriever" or "colbert" """
    backend = DENSE_BACKENDS[dense_backend]
    doc_text_map = _get_doc_text_map()

    bm25_hits = opensearch_utils.bm25_search(os_client, query_text)
    dense_hits = backend.search(query_text, id_to_text=doc_text_map)

    union = _union(bm25_hits, dense_hits)
    top_k_results = reranker.rerank(query_text, union, top_k=top_k)

    return {
        "bm25_count": len(bm25_hits),
        "dense_count": len(dense_hits),
        "union_count": len(union),
        "top_k": top_k_results,
    }
