from src.retrieval.retriever import LaborLawRetriever
from src.rag.context_builder import build_context
from src.rag.prompt_builder import build_prompt


retriever = LaborLawRetriever()

query = "كم مدة فترة التجربة؟"

results = retriever.search(query, top_k=5)

context = build_context(results, top_k=3)

prompt = build_prompt(query, context)

print(prompt)
