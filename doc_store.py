import json
import os
import config


def _path():
    return os.path.join(config.INDEX_DIR, "doc_store.json")


def build(docs):
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    mapping = {d["id"]: d["text"] for d in docs}
    with open(_path(), "w") as f:
        json.dump(mapping, f)
    print(f"[doc_store] saved {len(mapping)} docs -> {_path()}")


def load():
    path = _path()
    if not os.path.exists(path):
        raise FileNotFoundError("No doc store found. Run `python build_dense_index.py` first.")
    with open(path) as f:
        return json.load(f)