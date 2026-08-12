"""
Dense retrieval using Promptriever Llama3.1 8B (samaya-ai/promptriever-llama3.1-8b-v1).

This is a plain bi-encoder: one embedding per doc, one per query. We encode the
whole corpus ONCE (build_index, run via build_dense_index.py) and cache it to
disk, so query-time search is just an embed + cosine-sim lookup.

Needs a real GPU — run build_index on the EC2 L4 box, not the Mac.
"""
import os
import numpy as np
import mteb
import config

_model = None


def get_model():
    global _model
    if _model is None:
        _model = mteb.get_model(config.PROMPTRIEVER_MODEL)
    return _model


def _index_paths():
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    return (
        os.path.join(config.INDEX_DIR, "promptriever_embeddings.npy"),
        os.path.join(config.INDEX_DIR, "promptriever_ids.npy"),
    )


def build_index(docs):
    """docs: list of {"id": ..., "text": ...}. Encodes and caches to disk."""
    model = get_model()
    texts = [d["text"] for d in docs]
    ids = np.array([d["id"] for d in docs])

    embeddings = np.asarray(
        model.encode(texts, batch_size=config.PROMPTRIEVER_BATCH_SIZE, show_progress_bar=True)
    )

    emb_path, id_path = _index_paths()
    np.save(emb_path, embeddings)
    np.save(id_path, ids)
    print(f"[promptriever] indexed {len(ids)} docs -> {emb_path}")


def load_index():
    emb_path, id_path = _index_paths()
    if not (os.path.exists(emb_path) and os.path.exists(id_path)):
        raise FileNotFoundError(
            "No Promptriever index found. Run `python build_dense_index.py --model promptriever` first."
        )
    return np.load(emb_path), np.load(id_path)


def search(query_text, top_n=None, id_to_text=None):
    top_n = top_n or config.DENSE_TOP_N
    model = get_model()
    doc_embeddings, doc_ids = load_index()

    query_embedding = np.asarray(model.encode([query_text]))[0]

    doc_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    q_norm = query_embedding / np.linalg.norm(query_embedding)
    scores = doc_norm @ q_norm

    top_idx = np.argsort(-scores)[:top_n]
    id_to_text = id_to_text or {}
    return [
        {
            "id": str(doc_ids[i]),
            "text": id_to_text.get(str(doc_ids[i]), ""),
            "score": float(scores[i]),
        }
        for i in top_idx
    ]
