import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"

INDEX_PATH = Path(
    "data/index/labor_law_bge_m3.faiss"
)

METADATA_PATH = Path(
    "data/index/labor_law_metadata.json"
)


class LaborLawRetriever:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)

        print("Loading FAISS index...")
        self.index = faiss.read_index(
            str(INDEX_PATH)
        )

        print("Loading metadata...")
        self.documents = json.loads(
            METADATA_PATH.read_text(
                encoding="utf-8"
            )
        )

        if self.index.ntotal != len(self.documents):
            raise ValueError(
                "FAISS index and metadata size mismatch"
            )

    def search(self, query, top_k=5):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for rank, (idx, score) in enumerate(
            zip(indices[0], scores[0]),
            start=1,
        ):
            doc = self.documents[idx]

            results.append({
                "rank": rank,
                "score": float(score),
                "article_heading": doc[
                    "article_heading"
                ],
                "text_verbatim": doc[
                    "text_verbatim"
                ],
                "source_id": doc[
                    "source_id"
                ],
                "document_title": doc[
                    "document_title"
                ],
                "footnotes": doc[
                    "footnotes"
                ],
            })

        return results
