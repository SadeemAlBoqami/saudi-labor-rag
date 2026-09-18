from src.retrieval.retriever import LaborLawRetriever
from src.rag.context_builder import build_context
from src.rag.prompt_builder import build_prompt
from src.rag.generator import QwenGenerator


query = "كم مدة فترة التجربة؟"

print("Loading retriever...")
retriever = LaborLawRetriever()

print("Retrieving legal context...")
results = retriever.search(
    query,
    top_k=3,
)

context = build_context(
    results,
    top_k=3,
)

prompt = build_prompt(
    query,
    context,
)

print("Loading generator...")
generator = QwenGenerator()

print()
print("Generating answer...")
print()

answer = generator.generate(
    prompt,
    max_new_tokens=256,
)

print("=" * 70)
print("QUESTION:")
print(query)

print()
print("ANSWER:")
print(answer)
print("=" * 70)
