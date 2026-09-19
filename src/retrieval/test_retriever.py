from src.retrieval.retriever import LaborLawRetriever

retriever = LaborLawRetriever()

query = "كم مدة فترة التجربة؟"

results = retriever.search(
    query,
    top_k=5,
)

print()
print(f"Query: {query}")
print()

for result in results:
    print(
        f"{result['rank']}. "
        f"{result['article_heading']} "
        f"(score={result['score']:.4f})"
    )

    print(
        result["text_verbatim"][:300]
    )

    print("-" * 70)
