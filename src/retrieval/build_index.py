import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DOCUMENTS_FILE = Path(
    "data/processed/labor_law_documents.json"
)

INDEX_DIR = Path("data/index")

MODEL_NAME = "BAAI/bge-m3"


def main():
    print("Loading documents...")

    documents = json.loads(
        DOCUMENTS_FILE.read_text(encoding="utf-8")
    )

    texts = [
        doc["text_normalized"]
        for doc in documents
    ]

    print(f"Documents: {len(texts)}")

    print("Loading BGE-M3...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=4,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print("Building FAISS index...")

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    index_path = INDEX_DIR / "labor_law_bge_m3.faiss"

    metadata_path = (
        INDEX_DIR / "labor_law_metadata.json"
    )

    faiss.write_index(
        index,
        str(index_path)
    )

    metadata_path.write_text(
        json.dumps(
            documents,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("✓ Vector index created")
    print(f"✓ Vectors: {index.ntotal}")
    print(f"✓ Dimension: {embeddings.shape[1]}")
    print(f"✓ Index: {index_path}")
    print(f"✓ Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
