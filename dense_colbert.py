"""
Dense (multi-vector) retrieval using GTE-ModernColBERT (lightonai/GTE-ModernColBERT-v1).

Unlike Promptriever, this is a late-interaction model: it produces one
embedding PER TOKEN and scores with MaxSim, not a single cosine-sim. That's
architecturally different from a bi-encoder, so it gets its own index
(pylate's Voyager index) and its own search path.

pip install pylate
"""
import os
import config
from pylate import indexes, models, retrieve

_model = None
_retriever = None


def get_model():
    global _model
    if _model is None:
        _model = models.ColBERT(model_name_or_path=config.COLBERT_MODEL)
    return _model


def _index_path():
    return os.path.join(config.INDEX_DIR, "colbert_index")


def build_index(docs):
    """docs: list of {"id": ..., "text": ...}."""
    model = get_model()
    os.makedirs(config.INDEX_DIR, exist_ok=True)

    index = indexes.Voyager(
        index_folder=_index_path(),
        index_name="gte_moderncolbert",
        override=True,
    )
    doc_ids = [d["id"] for d in docs]
    doc_texts = [d["text"] for d in docs]
    doc_embeddings = model.encode(
        doc_texts, batch_size=16, is_query=False, show_progress_bar=True
    )
    index.add_documents(documents_ids=doc_ids, documents_embeddings=doc_embeddings)
    print(f"[gte-moderncolbert] indexed {len(doc_ids)} docs -> {_index_path()}")


def get_retriever():
    global _retriever
    if _retriever is None:
        index = indexes.Voyager(
            index_folder=_index_path(),
            index_name="gte_moderncolbert",
        )
        _retriever = retrieve.ColBERT(index=index)
    return _retriever


def search(query_text, top_n=None, id_to_text=None):
    top_n = top_n or config.DENSE_TOP_N
    model = get_model()
    retriever = get_retriever()

    query_embedding = model.encode([query_text], is_query=True)
    hits = retriever.retrieve(queries_embeddings=query_embedding, k=top_n)[0]

    id_to_text = id_to_text or {}
    return [
        {
            "id": str(h["id"]),
            "text": id_to_text.get(str(h["id"]), ""),
            "score": float(h["score"]),
        }
        for h in hits
    ]
