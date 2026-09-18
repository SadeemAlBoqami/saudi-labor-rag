## Embedding Layer

The embedding layer converts each legal article and user query into numerical vectors so the system can retrieve the most semantically relevant articles before sending them to the LLM.

This is the retrieval component of the RAG pipeline.

### Models Evaluated

- **Qwen3-Embedding-0.6B**
  - Multilingual embedding model.
  - Worked correctly, but was relatively heavy on CPU in the development environment.
  - Excluded from the full benchmark because the project targets a resource-constrained Jetson device.

- **multilingual-e5-large**
  - Strong multilingual retrieval model.
  - Achieved high retrieval accuracy on the Arabic legal test set.

- **BGE-M3**
  - Multilingual embedding model designed for retrieval.
  - Achieved the best overall retrieval performance in our benchmark.

### Why BGE-M3?

On the 45-question Saudi Labor Law evaluation set:

- Recall@1: **95.56%**
- Recall@3: **100%**
- Recall@5: **100%**
- MRR: **0.9704**
- Average query latency: **~340 ms**

BGE-M3 was selected because it matched E5 at Recall@1, achieved better Recall@3 and MRR, and had lower query latency.

**Selected embedding model: `BAAI/bge-m3`**