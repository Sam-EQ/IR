"""
Everything OpenSearch-related: connecting, BM25 candidate retrieval (shared by
both flows), and pulling the full corpus once to build the dense indexes.
"""
from opensearchpy import OpenSearch
import config


def get_client():
    return OpenSearch(
        hosts=[config.OPENSEARCH_HOST],
        http_auth=(config.OPENSEARCH_USER, config.OPENSEARCH_PASS),
        use_ssl=True,
        verify_certs=True,
    )


def fetch_all_docs(client, index=None, page_size=500):
    """Scroll through every doc in the index. Run once, at index-build time."""
    index = index or config.OPENSEARCH_INDEX
    docs = []
    resp = client.search(
        index=index,
        body={"query": {"match_all": {}}, "size": page_size},
        scroll="2m",
    )
    scroll_id = resp["_scroll_id"]
    hits = resp["hits"]["hits"]
    while hits:
        for h in hits:
            docs.append({"id": h["_id"], "text": h["_source"].get(config.TEXT_FIELD, "")})
        resp = client.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]
    client.clear_scroll(scroll_id=scroll_id)
    return docs


def bm25_search(client, query_text, top_n=None, index=None):
    """The BM25 leg that feeds into both flows."""
    index = index or config.OPENSEARCH_INDEX
    top_n = top_n or config.BM25_TOP_N
    resp = client.search(
        index=index,
        body={
            "query": {"match": {config.TEXT_FIELD: query_text}},
            "size": top_n,
        },
    )
    return [
        {"id": h["_id"], "text": h["_source"].get(config.TEXT_FIELD, ""), "score": h["_score"]}
        for h in resp["hits"]["hits"]
    ]
