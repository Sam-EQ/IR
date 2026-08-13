import argparse
import opensearch_utils
import pipeline


def print_result(label, result):
    print(f"\n--- {label} ---")
    print(
        f"BM25 candidates: {result['bm25_count']} | "
        f"dense candidates: {result['dense_count']} | "
        f"union: {result['union_count']}"
    )
    for rank, doc in enumerate(result["top_k"], 1):
        snippet = doc["text"][:180].replace("\n", " ")
        print(f"  {rank}. [{doc['id']}] {snippet}")


def compare_query(query_text, os_client):
    print(f"\n================ QUERY: {query_text} ================")

    result_a = pipeline.run_pipeline(query_text, "promptriever", os_client)
    print_result("Flow A: BM25 + Promptriever Llama3 8B", result_a)

    result_b = pipeline.run_pipeline(query_text, "colbert", os_client)
    print_result("Flow B: BM25 + GTE-ModernColBERT", result_b)

    ids_a = {d["id"] for d in result_a["top_k"]}
    ids_b = {d["id"] for d in result_b["top_k"]}
    both = ids_a & ids_b
    only_a = ids_a - ids_b
    only_b = ids_b - ids_a

    print(f"\n  overlap: {len(both)}/{len(ids_a | ids_b)} docs shared between the two flows")
    print(f"  only in Flow A (Promptriever): {sorted(only_a)}")
    print(f"  only in Flow B (GTE-ModernColBERT): {sorted(only_b)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="a single query to test")
    parser.add_argument("--file", help="path to a text file, one query per line")
    args = parser.parse_args()

    if not args.query and not args.file:
        parser.error("give a query string or --file")

    os_client = opensearch_utils.get_client()

    queries = (
        [args.query]
        if args.query
        else [line.strip() for line in open(args.file) if line.strip()]
    )

    for q in queries:
        compare_query(q, os_client)


if __name__ == "__main__":
    main()