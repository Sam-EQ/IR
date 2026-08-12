"""
Reranks a candidate pool with Gemini 2.5 Pro — the reranker both flows specify.
Feeds the whole candidate set + query in one prompt and asks for a ranked
doc-id list back (same long-context rerank approach the LIMIT paper used to
test Gemini-2.5-Pro as a reranker).

pip install google-generativeai
"""
import json
import re
import google.generativeai as genai
import config

genai.configure(api_key=config.GEMINI_API_KEY)

_PROMPT = """You are a relevance ranking system.

Query: {query}

Below are candidate documents, each with an ID. Rank ALL of them from most to
least relevant to the query. Return ONLY a JSON array of document IDs, most
relevant first. No other text, no markdown fences.

Documents:
{docs}
"""


def _format_docs(candidates):
    lines = []
    for c in candidates:
        snippet = c["text"][:1000].replace("\n", " ")
        lines.append(f'[{c["id"]}] {snippet}')
    return "\n".join(lines)


def rerank(query_text, candidates, top_k=None):
    """candidates: list of {"id", "text", "score"} — the deduped BM25+dense union."""
    top_k = top_k or config.RERANK_TOP_K
    if not candidates:
        return []

    model = genai.GenerativeModel(config.GEMINI_RERANK_MODEL)
    prompt = _PROMPT.format(query=query_text, docs=_format_docs(candidates))
    response = model.generate_content(prompt)

    raw = (response.text or "").strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    ranked_ids = json.loads(match.group(0)) if match else []

    by_id = {c["id"]: c for c in candidates}
    ranked = [by_id[str(i)] for i in ranked_ids if str(i) in by_id]

    # anything Gemini didn't mention stays at the end, in its original order
    seen = {r["id"] for r in ranked}
    for c in candidates:
        if c["id"] not in seen:
            ranked.append(c)

    return ranked[:top_k]
