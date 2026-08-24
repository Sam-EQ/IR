import argparse
import sys
import opensearch_utils
import pipeline


def print_docs(result, out):
    for rank, doc in enumerate(result["top_k"], 1):
        print(f"\nRank {rank} — [{doc['id']}] (score: {doc['score']:.4f})", file=out)
        print(doc["text"], file=out)
        print("-" * 80, file=out)


def print_analysis(result_a, result_b, out):
    ids_a = [d["id"] for d in result_a["top_k"]]
    ids_b = [d["id"] for d in result_b["top_k"]]

    set_a = set(ids_a)
    set_b = set(ids_b)
    both = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a

    print("\n=== ANALYSIS ===", file=out)
    print(
        f"BM25 candidates: A={result_a['bm25_count']} B={result_b['bm25_count']} | "
        f"dense candidates: A={result_a['dense_count']} B={result_b['dense_count']} | "
        f"union: A={result_a['union_count']} B={result_b['union_count']}",
        file=out,
    )

    if ids_a:
        top_a = ids_a[0]
        pos_in_b = ids_b.index(top_a) + 1 if top_a in ids_b else None
        where = f"rank {pos_in_b} in Flow B" if pos_in_b else "not in Flow B's top-10"
        print(f"Flow A's #1 pick: [{top_a}] — {where}", file=out)

    if ids_b:
        top_b = ids_b[0]
        pos_in_a = ids_a.index(top_b) + 1 if top_b in ids_a else None
        where = f"rank {pos_in_a} in Flow A" if pos_in_a else "not in Flow A's top-10"
        print(f"Flow B's #1 pick: [{top_b}] — {where}", file=out)

    print(f"\nOverlap: {len(both)}/{len(set_a | set_b)} docs shared between the two flows", file=out)
    print(f"Only in Flow A (Promptriever): {sorted(only_a)}", file=out)
    print(f"Only in Flow B (GTE-ModernColBERT): {sorted(only_b)}", file=out)


def compare_query(query_text, os_client, out):
    print(f"\n{'=' * 100}", file=out)
    print(f"QUERY: {query_text}", file=out)
    print(f"{'=' * 100}", file=out)

    result_a = pipeline.run_pipeline(query_text, "promptriever", os_client)
    result_b = pipeline.run_pipeline(query_text, "colbert", os_client)

    print("\n### Flow A: BM25 + Promptriever Llama3 8B ###", file=out)
    print_docs(result_a, out)

    print("\n### Flow B: BM25 + GTE-ModernColBERT ###", file=out)
    print_docs(result_b, out)

    print_analysis(result_a, result_b, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="a single query to test")
    parser.add_argument("--file", help="path to a text file, one query per line")
    parser.add_argument("--output", help="write results to this file instead of the terminal")
    args = parser.parse_args()

    if not args.query and not args.file:
        parser.error("give a query string or --file")

    os_client = opensearch_utils.get_client()

    queries = (
        [args.query]
        if args.query
        else [line.strip() for line in open(args.file) if line.strip()]
    )

    out = open(args.output, "w") if args.output else sys.stdout

    for q in queries:
        compare_query(q, os_client, out)

    if args.output:
        out.close()
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()