import json
import re
import time
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
import config

_client = OpenAI(api_key=config.OPENAI_API_KEY)

_PROMPT = """You are a relevance ranking system.

Query: {query}

Below are candidate documents, each with an ID. Rank ALL of them from most to
least relevant to the query. Return ONLY a JSON array of document IDs, most
relevant first. No other text, no markdown fences.

Documents:
{docs}
"""

_MAX_RETRIES = 4
_BASE_DELAY_SECONDS = 5


def _format_docs(candidates):
    lines = []
    for c in candidates:
        snippet = c["text"][:1000].replace("\n", " ")
        lines.append(f'[{c["id"]}] {snippet}')
    return "\n".join(lines)


def _generate_with_retry(prompt):
    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            return _client.chat.completions.create(
                model=config.OPENAI_RERANK_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
        except (RateLimitError, APIConnectionError, APIError) as e:
            last_error = e
            delay = _BASE_DELAY_SECONDS * (2 ** attempt)
            print(f"[reranker] OpenAI call failed ({e}), retrying in {delay}s...")
            time.sleep(delay)
    raise last_error


def rerank(query_text, candidates, top_k=None):
    top_k = top_k or config.RERANK_TOP_K
    if not candidates:
        return []

    prompt = _PROMPT.format(query=query_text, docs=_format_docs(candidates))
    response = _generate_with_retry(prompt)

    raw = (response.choices[0].message.content or "").strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    ranked_ids = json.loads(match.group(0)) if match else []

    by_id = {c["id"]: c for c in candidates}
    ranked = [by_id[str(i)] for i in ranked_ids if str(i) in by_id]

    seen = {r["id"] for r in ranked}
    for c in candidates:
        if c["id"] not in seen:
            ranked.append(c)

    return ranked[:top_k]