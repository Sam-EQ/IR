import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
import config

QUERY_PREFIX = "query:  "
PASSAGE_PREFIX = "passage:  "

_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    peft_config = PeftConfig.from_pretrained(config.PROMPTRIEVER_MODEL)
    base_model_name = peft_config.base_model_name_or_path

    base_model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = PeftModel.from_pretrained(base_model, config.PROMPTRIEVER_MODEL)
    model.config.max_length = 512
    tokenizer.model_max_length = 512

    model.eval()
    _model, _tokenizer = model, tokenizer
    return _model, _tokenizer


def _batch_dict(tokenizer, texts, max_length):
    batch_dict = tokenizer(
        texts,
        max_length=max_length - 1,
        return_token_type_ids=False,
        return_attention_mask=False,
        padding=False,
        truncation=True,
    )
    batch_dict["input_ids"] = [ids + [tokenizer.eos_token_id] for ids in batch_dict["input_ids"]]
    return tokenizer.pad(
        batch_dict,
        padding=True,
        pad_to_multiple_of=8,
        return_attention_mask=True,
        return_tensors="pt",
    )


def _encode(texts, batch_size=None):
    model, tokenizer = _load()
    batch_size = batch_size or config.PROMPTRIEVER_BATCH_SIZE
    max_length = model.config.max_length

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_dict = _batch_dict(tokenizer, batch_texts, max_length)
        batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = model(**batch_dict)
            last_hidden = outputs.last_hidden_state
            seq_lens = batch_dict["attention_mask"].sum(dim=1) - 1
            bsz = last_hidden.shape[0]
            reps = last_hidden[torch.arange(bsz, device=last_hidden.device), seq_lens]
            embeddings = F.normalize(reps, p=2, dim=-1)
            all_embeddings.append(embeddings.float().cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def _index_paths():
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    return (
        os.path.join(config.INDEX_DIR, "promptriever_embeddings.npy"),
        os.path.join(config.INDEX_DIR, "promptriever_ids.npy"),
    )


def build_index(docs):
    texts = [PASSAGE_PREFIX + d["text"] for d in docs]
    ids = np.array([d["id"] for d in docs])

    embeddings = _encode(texts)

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
    doc_embeddings, doc_ids = load_index()

    query_embedding = _encode([QUERY_PREFIX + query_text])[0]

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