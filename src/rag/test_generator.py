from src.retrieval.retriever import LaborLawRetriever
from src.rag.context_builder import build_context
from src.rag.prompt_builder import build_prompt
from src.rag.generator import QwenGenerator


query = "كم مدة فترة التجربة؟"

retriever = LaborLawRetriever()

results = retriever.search(
    query,
    top_k=1,
)

context = build_context(
    results,
    top_k=1,
)

prompt = build_prompt(
    query,
    context,
)

generator = QwenGenerator()

answer = generator.generate(
    prompt,
    max_new_tokens=256,
)

sources = [
    result["article_heading"]
    for result in results
]

print("=" * 70)
print("QUESTION:")
print(query)

print()
print("ANSWER:")
print(answer)

print()
print("SOURCES:")
for source in sources:
    print(f"- {source}")

print("=" * 70)