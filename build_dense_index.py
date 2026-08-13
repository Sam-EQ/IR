import argparse
import opensearch_utils
import doc_store
import dense_promptriever
import dense_colbert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["promptriever", "colbert", "both"], default="both")
    args = parser.parse_args()

    client = opensearch_utils.get_client()
    docs = opensearch_utils.fetch_all_docs(client)
    print(f"Fetched {len(docs)} docs from OpenSearch index.")

    doc_store.build(docs)

    if args.model in ("promptriever", "both"):
        dense_promptriever.build_index(docs)
    if args.model in ("colbert", "both"):
        dense_colbert.build_index(docs)


if __name__ == "__main__":
    main()