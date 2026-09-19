import json
import re

from src.retrieval.retriever import LaborLawRetriever
from src.rag.context_builder import build_context
from src.rag.prompt_builder import build_prompt
from src.rag.generator import QwenGenerator


# ============================================================
# User-facing messages
# ============================================================

OUT_OF_SCOPE_MESSAGE = (
    "لا توجد معلومات موثوقة كفاية في نظام العمل المسترجع "
    "تسمح بالإجابة عن هذا السؤال."
)

LOW_CONFIDENCE_MESSAGE = (
    "لم أتمكن من تحديد نص نظامي ذي صلة بدرجة ثقة كافية "
    "للإجابة بدقة."
)


# ============================================================
# Retrieval configuration
# ============================================================

RETRIEVE_K = 3


# ============================================================
# Safety thresholds
# ============================================================

# أقل من هذا:
# نعتبر الاسترجاع ضعيفاً جداً ونرفض مباشرة.
HARD_REJECT_SCORE = 0.05

# من 0.05 إلى أقل من 0.15:
# Gray zone
# لا نسمح فيها بأي إجابة قانونية مباشرة.
GRAY_ZONE_MAX_SCORE = 0.15

# من 0.15 فأعلى:
# يسمح بتمرير السؤال إلى Gemma كمرشح قوي.
STRONG_ACCEPT_SCORE = 0.15


class LaborLawRAG:
    """
    Safety-first Saudi Labor Law RAG pipeline.

    Pipeline:

        Query
          ↓
        BGE-M3 dense retrieval
          ↓
        FAISS Top-3
          ↓
        Cross-encoder reranking
          ↓
        Safety score gate
          ↓
        Best legal article only
          ↓
        Gemma
          ↓
        answer / clarify / out_of_scope

    Design principle:

        Prefer abstention over a potentially incorrect
        legal answer.
    """

    def __init__(self):
        print("Loading Labor Law RAG pipeline...")

        self.retriever = LaborLawRetriever()
        self.generator = QwenGenerator()

        print("✓ Safety-first RAG pipeline ready")

    # ========================================================
    # JSON parsing
    # ========================================================

    @staticmethod
    def _parse_json_output(raw_output: str):
        """
        Parse structured model output safely.

        Supports:
        - plain JSON
        - ```json ... ```
        - ``` ... ```
        - extra surrounding text
        """

        if not raw_output:
            return None

        text = raw_output.strip()

        text = re.sub(
            r"^```json\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"^```\s*",
            "",
            text,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        text = text.strip()

        # Direct JSON parse
        try:
            return json.loads(text)

        except json.JSONDecodeError:
            pass

        # Extract first JSON object
        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL,
        )

        if not match:
            return None

        try:
            return json.loads(
                match.group(0)
            )

        except json.JSONDecodeError:
            return None

    # ========================================================
    # Score zone classifier
    # ========================================================

    @staticmethod
    def _score_zone(best_score: float):
        """
        Classify reranker score into one of:

        - reject
        - gray
        - strong
        """

        if best_score < HARD_REJECT_SCORE:
            return "reject"

        if best_score < GRAY_ZONE_MAX_SCORE:
            return "gray"

        return "strong"

    # ========================================================
    # Debug helper
    # ========================================================

    @staticmethod
    def _build_debug_fields(
        dense_candidates: list,
        reranked_results: list,
    ):
        candidate_debug = []

        for item in reranked_results:
            candidate_debug.append(
                {
                    "article_heading": item.get(
                        "article_heading",
                        "",
                    ),
                    "dense_score": item.get(
                        "dense_score",
                        None,
                    ),
                    "reranker_score": item.get(
                        "reranker_score",
                        None,
                    ),
                    "reranker_confidence": item.get(
                        "reranker_confidence",
                        None,
                    ),
                    "dense_rank": item.get(
                        "rank",
                        None,
                    ),
                    "rerank_rank": item.get(
                        "rerank_rank",
                        None,
                    ),
                }
            )

        return {
            "retrieval_k": len(dense_candidates),
            "retrieval_candidates": candidate_debug,
        }

    # ========================================================
    # Diagnostic fields
    # ========================================================

    @staticmethod
    def _diagnostic_fields(
        best_candidate,
        reranked_results,
        score_zone,
    ):
        best_dense_score = 0.0
        best_reranker_score = 0.0
        best_reranker_confidence = 0.0
        reranker_margin = None

        if best_candidate:
            best_dense_score = float(
                best_candidate.get(
                    "dense_score",
                    0.0,
                )
            )

            best_reranker_score = float(
                best_candidate.get(
                    "reranker_score",
                    0.0,
                )
            )

            best_reranker_confidence = float(
                best_candidate.get(
                    "reranker_confidence",
                    0.0,
                )
            )

        if (
            reranked_results
            and len(reranked_results) >= 2
        ):
            first_score = float(
                reranked_results[0].get(
                    "reranker_score",
                    0.0,
                )
            )

            second_score = float(
                reranked_results[1].get(
                    "reranker_score",
                    0.0,
                )
            )

            reranker_margin = (
                first_score
                - second_score
            )

        return {
            "retrieval_score": best_dense_score,
            "reranker_score": best_reranker_score,
            "reranker_confidence": best_reranker_confidence,
            "reranker_margin": reranker_margin,
            "score_zone": score_zone,
        }

    # ========================================================
    # Main pipeline
    # ========================================================

    def answer(
        self,
        query: str,
    ):
        query = query.strip()

        # ====================================================
        # 0. Empty query
        # ====================================================

        if not query:
            return {
                "status": "error",
                "answer": "",
                "clarifying_question": "",
                "sources": [],
                "retrieval_k": 0,
                "retrieval_score": 0.0,
                "reranker_score": 0.0,
                "reranker_confidence": 0.0,
                "reranker_margin": None,
                "score_zone": "reject",
                "retrieval_candidates": [],
                "abstained": True,
                "abstention_reason": "empty_query",
            }

        # ====================================================
        # 1. Dense Top-3 retrieval
        # ====================================================

        dense_candidates = (
            self.retriever.search(
                query,
                top_k=RETRIEVE_K,
            )
        )

        if not dense_candidates:
            return {
                "status": "out_of_scope",
                "answer": OUT_OF_SCOPE_MESSAGE,
                "clarifying_question": "",
                "sources": [],
                "retrieval_k": 0,
                "retrieval_score": 0.0,
                "reranker_score": 0.0,
                "reranker_confidence": 0.0,
                "reranker_margin": None,
                "score_zone": "reject",
                "retrieval_candidates": [],
                "abstained": True,
                "abstention_reason": "no_retrieval_candidates",
            }

        # ====================================================
        # 2. Rerank Top-3
        # ====================================================

        reranked_results = (
            self.retriever.rerank(
                query,
                dense_candidates,
            )
        )

        debug_fields = (
            self._build_debug_fields(
                dense_candidates,
                reranked_results,
            )
        )

        if not reranked_results:
            return {
                "status": "out_of_scope",
                "answer": LOW_CONFIDENCE_MESSAGE,
                "clarifying_question": "",
                "sources": [],
                "retrieval_score": 0.0,
                "reranker_score": 0.0,
                "reranker_confidence": 0.0,
                "reranker_margin": None,
                "score_zone": "reject",
                "abstained": True,
                "abstention_reason": "no_reranked_candidates",
                **debug_fields,
            }

        # ====================================================
        # 3. Best candidate
        # ====================================================

        best_candidate = (
            reranked_results[0]
        )

        best_reranker_score = float(
            best_candidate.get(
                "reranker_score",
                0.0,
            )
        )

        score_zone = (
            self._score_zone(
                best_reranker_score
            )
        )

        diagnostics = (
            self._diagnostic_fields(
                best_candidate,
                reranked_results,
                score_zone,
            )
        )

        # ====================================================
        # 4. HARD REJECT
        # ====================================================

        if score_zone == "reject":
            return {
                "status": "out_of_scope",
                "answer": LOW_CONFIDENCE_MESSAGE,
                "clarifying_question": "",
                "sources": [],
                "abstained": True,
                "abstention_reason": "reranker_score_too_low",
                **diagnostics,
                **debug_fields,
            }

        # ====================================================
        # 5. Select ONE best article only
        # ====================================================

        selected_result = best_candidate

        selected_results = [
            selected_result
        ]

        article_heading = (
            selected_result.get(
                "article_heading",
                "",
            )
        )

        sources = (
            [article_heading]
            if article_heading
            else []
        )

        # ====================================================
        # 6. Build context
        # ====================================================

        context = build_context(
            selected_results
        )

        # ====================================================
        # 7. Build prompt
        # ====================================================

        prompt = build_prompt(
            query,
            context,
        )

        # ====================================================
        # 8. Generate with Gemma
        # ====================================================

        raw_output = (
            self.generator.generate(
                prompt,
                max_new_tokens=512,
            )
        )

        print()
        print("--- RAW MODEL OUTPUT ---")
        print(raw_output)
        print("--- END RAW OUTPUT ---")
        print()

        # ====================================================
        # 9. Parse JSON
        # ====================================================

        parsed = (
            self._parse_json_output(
                raw_output
            )
        )

        if not parsed:
            return {
                "status": "error",
                "answer": "",
                "clarifying_question": "",
                "sources": sources,
                "abstained": True,
                "abstention_reason": "json_parse_failure",
                **diagnostics,
                **debug_fields,
            }

        # ====================================================
        # 10. Extract model fields
        # ====================================================

        status = str(
            parsed.get(
                "status",
                "",
            )
        ).strip()

        answer = str(
            parsed.get(
                "answer",
                "",
            )
            or ""
        ).strip()

        clarifying_question = str(
            parsed.get(
                "clarifying_question",
                "",
            )
            or ""
        ).strip()

        valid_statuses = {
            "answer",
            "clarify",
            "out_of_scope",
        }

        if status not in valid_statuses:
            return {
                "status": "error",
                "answer": "",
                "clarifying_question": "",
                "sources": sources,
                "abstained": True,
                "abstention_reason": "invalid_model_status",
                **diagnostics,
                **debug_fields,
            }

        # ====================================================
        # 11. GRAY ZONE SAFETY POLICY
        # ====================================================

        if score_zone == "gray":
            return {
                "status": "out_of_scope",
                "answer": LOW_CONFIDENCE_MESSAGE,
                "clarifying_question": "",
                "sources": [],
                "abstained": True,
                "abstention_reason": "gray_zone_abstention",
                **diagnostics,
                **debug_fields,
            }

        # ====================================================
        # 12. STRONG ZONE
        # ====================================================

        # ---------------------------------------------
        # ANSWER
        # ---------------------------------------------

        if status == "answer":
            if not answer:
                return {
                    "status": "out_of_scope",
                    "answer": LOW_CONFIDENCE_MESSAGE,
                    "clarifying_question": "",
                    "sources": [],
                    "abstained": True,
                    "abstention_reason": "empty_model_answer",
                    **diagnostics,
                    **debug_fields,
                }

            return {
                "status": "answer",
                "answer": answer,
                "clarifying_question": "",
                "sources": sources,
                "abstained": False,
                "abstention_reason": None,
                **diagnostics,
                **debug_fields,
            }

        # ---------------------------------------------
        # CLARIFY
        # ---------------------------------------------

        if status == "clarify":
            if not clarifying_question:
                return {
                    "status": "out_of_scope",
                    "answer": LOW_CONFIDENCE_MESSAGE,
                    "clarifying_question": "",
                    "sources": [],
                    "abstained": True,
                    "abstention_reason": "empty_clarification",
                    **diagnostics,
                    **debug_fields,
                }

            return {
                "status": "clarify",
                "answer": "",
                "clarifying_question": clarifying_question,
                "sources": sources,
                "abstained": False,
                "abstention_reason": None,
                **diagnostics,
                **debug_fields,
            }

        # ---------------------------------------------
        # OUT OF SCOPE
        # ---------------------------------------------

        return {
            "status": "out_of_scope",
            "answer": OUT_OF_SCOPE_MESSAGE,
            "clarifying_question": "",
            "sources": [],
            "abstained": True,
            "abstention_reason": "model_out_of_scope",
            **diagnostics,
            **debug_fields,
        }