import math
from pathlib import Path

import faiss
import json
import numpy as np

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)


# ============================================================
# Configuration
# ============================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

RERANKER_MODEL_NAME = (
    "BAAI/bge-reranker-v2-m3"
)

INDEX_PATH = Path(
    "data/index/labor_law_bge_m3.faiss"
)

METADATA_PATH = Path(
    "data/index/labor_law_metadata.json"
)


class LaborLawRetriever:
    """
    Safety-oriented retrieval pipeline.

    Stage 1:
        Dense retrieval using:
        BAAI/bge-m3 + FAISS

    Stage 2:
        Cross-encoder reranking using:
        BAAI/bge-reranker-v2-m3

    The retriever first retrieves a wider candidate set,
    then reranks candidates according to query-document
    relevance.

    It does NOT generate legal answers.
    """

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        reranker_model_name: str = RERANKER_MODEL_NAME,
    ):
        print(
            f"Loading embedding model: "
            f"{embedding_model_name}"
        )

        self.embedding_model = (
            SentenceTransformer(
                embedding_model_name
            )
        )

        print(
            f"Loading reranker model: "
            f"{reranker_model_name}"
        )

        self.reranker = CrossEncoder(
            reranker_model_name
        )

        print(
            f"Loading FAISS index: "
            f"{INDEX_PATH}"
        )

        self.index = faiss.read_index(
            str(INDEX_PATH)
        )

        print(
            f"Loading retrieval metadata: "
            f"{METADATA_PATH}"
        )

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as f:
            self.metadata = json.load(f)

        # Some metadata files may be:
        #
        # [
        #   {...},
        #   {...}
        # ]
        #
        # while others may contain:
        #
        # {
        #   "documents": [...]
        # }
        #
        # Support both.

        if isinstance(
            self.metadata,
            dict,
        ):
            if "documents" in self.metadata:
                self.metadata = (
                    self.metadata["documents"]
                )
            elif "metadata" in self.metadata:
                self.metadata = (
                    self.metadata["metadata"]
                )

        if not isinstance(
            self.metadata,
            list,
        ):
            raise ValueError(
                "Retrieval metadata must "
                "contain a list of documents."
            )

        if (
            self.index.ntotal
            != len(self.metadata)
        ):
            raise ValueError(
                "FAISS index size does not "
                "match metadata size: "
                f"{self.index.ntotal} vectors "
                f"vs {len(self.metadata)} "
                f"metadata records."
            )

        print(
            "✓ Retriever ready"
        )

        print(
            f"✓ Indexed documents: "
            f"{self.index.ntotal}"
        )

    # ========================================================
    # Dense retrieval
    # ========================================================

    def _encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Encode one query using BGE-M3.

        FAISS uses inner product on normalized vectors,
        therefore embeddings are normalized.
        """

        embedding = (
            self.embedding_model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        )

        embedding = np.asarray(
            embedding,
            dtype="float32",
        )

        return embedding

    def search(
        self,
        query: str,
        top_k: int = 3,
    ):
        """
        Dense FAISS retrieval.

        Returns the top_k candidates without reranking.
        """

        query = query.strip()

        if not query:
            return []

        query_embedding = (
            self._encode_query(
                query
            )
        )

        scores, indices = (
            self.index.search(
                query_embedding,
                top_k,
            )
        )

        results = []

        for rank, (
            index_id,
            score,
        ) in enumerate(
            zip(
                indices[0],
                scores[0],
            ),
            start=1,
        ):
            if index_id < 0:
                continue

            metadata = (
                self.metadata[
                    int(index_id)
                ]
            )

            result = {
                "rank":
                    rank,

                "index_id":
                    int(index_id),

                # Keep the original dense score.
                "score":
                    float(score),

                "dense_score":
                    float(score),

                "article_heading":
                    metadata.get(
                        "article_heading",
                        "",
                    ),

                "text_verbatim":
                    metadata.get(
                        "text_verbatim",
                        metadata.get(
                            "article_text",
                            "",
                        ),
                    ),

                "text_normalized":
                    metadata.get(
                        "text_normalized",
                        "",
                    ),

                "source_id":
                    metadata.get(
                        "source_id",
                        "labor_law",
                    ),

                "document_title":
                    metadata.get(
                        "document_title",
                        "نظام العمل",
                    ),

                "footnotes":
                    metadata.get(
                        "footnotes",
                        [],
                    ),
            }

            results.append(
                result
            )

        return results

    # ========================================================
    # Reranking
    # ========================================================

    @staticmethod
    def _sigmoid(
        value: float,
    ) -> float:
        """
        Convert an arbitrary reranker logit into a
        convenient 0-1 score.

        This is used only for confidence diagnostics.
        Ranking itself uses the raw reranker score.
        """

        # Protect against overflow.
        if value >= 0:
            z = math.exp(
                -value
            )

            return (
                1.0
                / (1.0 + z)
            )

        z = math.exp(
            value
        )

        return (
            z
            / (1.0 + z)
        )

    def rerank(
        self,
        query: str,
        candidates: list,
    ):
        """
        Rerank retrieved candidates using a
        cross-encoder.

        Cross-encoder input:

            [query, candidate legal text]

        Returns candidates ordered from most relevant
        to least relevant.
        """

        if not candidates:
            return []

        pairs = []

        for candidate in candidates:
            article_heading = (
                candidate.get(
                    "article_heading",
                    "",
                )
            )

            legal_text = (
                candidate.get(
                    "text_verbatim",
                    "",
                )
            )

            # Include the legal heading because article
            # identity may carry useful information.
            document_text = (
                f"{article_heading}\n"
                f"{legal_text}"
            )

            pairs.append(
                [
                    query,
                    document_text,
                ]
            )

        raw_scores = (
            self.reranker.predict(
                pairs,
                show_progress_bar=False,
            )
        )

        raw_scores = np.asarray(
            raw_scores
        ).reshape(-1)

        reranked = []

        for candidate, raw_score in zip(
            candidates,
            raw_scores,
        ):
            item = dict(
                candidate
            )

            raw_score = float(
                raw_score
            )

            item[
                "reranker_score"
            ] = raw_score

            item[
                "reranker_confidence"
            ] = self._sigmoid(
                raw_score
            )

            reranked.append(
                item
            )

        reranked.sort(
            key=lambda item: (
                item[
                    "reranker_score"
                ]
            ),
            reverse=True,
        )

        for rank, item in enumerate(
            reranked,
            start=1,
        ):
            item[
                "rerank_rank"
            ] = rank

        return reranked

    # ========================================================
    # Retrieve + rerank
    # ========================================================

    def retrieve_and_rerank(
        self,
        query: str,
        retrieve_k: int = 3,
    ):
        """
        Main retrieval interface.

        1. Retrieve Top-K with BGE-M3 + FAISS
        2. Rerank candidates with BGE reranker
        3. Return reranked candidates
        """

        dense_candidates = (
            self.search(
                query,
                top_k=retrieve_k,
            )
        )

        if not dense_candidates:
            return []

        return self.rerank(
            query,
            dense_candidates,
        )