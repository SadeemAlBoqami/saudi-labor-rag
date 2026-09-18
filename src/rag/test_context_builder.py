from src.retrieval.retriever import LaborLawRetriever
from src.rag.context_builder import build_context


retriever = LaborLawRetriever()

query = "كم مدة فترة التجربة؟"

results = retriever.search(
    query,
    top_k=5,
)

context = build_context(
    results,
    top_k=3,
)

print()
print("QUERY:")
print(query)

print()
print("CONTEXT:")
print("=" * 70)
print(context)
print("=" * 70)
