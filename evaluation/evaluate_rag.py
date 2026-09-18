import json
import re
import time
from pathlib import Path

from src.rag.rag_pipeline import LaborLawRAG


EVAL_FILE = Path(
    "evaluation/rag_eval.json"
)

OUTPUT_FILE = Path(
    "benchmarks/rag_factual_results.json"
)


# =====================================================
# Text normalization
# =====================================================

def normalize_text(text: str) -> str:
    """
    Lightweight Arabic normalization used only
    for evaluation matching.

    This DOES NOT modify the canonical legal corpus.
    """

    if not text:
        return ""

    text = text.strip()

    # Remove tatweel
    text = text.replace(
        "ـ",
        "",
    )

    # Normalize alef forms
    text = re.sub(
        r"[إأآا]",
        "ا",
        text,
    )

    # Normalize alef maqsura
    text = text.replace(
        "ى",
        "ي",
    )

    # -------------------------------------------------
    # Normalize common Arabic number case variants
    # -------------------------------------------------

    number_variants = {
        "عشرون": "عشرين",
        "ثلاثون": "ثلاثين",
        "اربعون": "اربعين",
        "خمسون": "خمسين",
        "ستون": "ستين",
        "سبعون": "سبعين",
        "ثمانون": "ثمانين",
        "تسعون": "تسعين",
    }

    for source, target in number_variants.items():
        text = text.replace(
            source,
            target,
        )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =====================================================
# Factual evaluation
# =====================================================

def evaluate_expected_facts(
    answer: str,
    expected_facts: list,
):
    """
    expected_facts format:

    [
        ["مائة وثمانين", "180"],
        ["ثلاثين", "30"]
    ]

    Each inner list represents acceptable alternatives.

    A fact is considered matched if ANY alternative
    appears in the generated answer.
    """

    if not expected_facts:
        return None, []

    normalized_answer = normalize_text(
        answer
    )

    fact_results = []

    for alternatives in expected_facts:
        matched = False
        matched_value = None

        for alternative in alternatives:
            normalized_alternative = (
                normalize_text(
                    alternative
                )
            )

            if (
                normalized_alternative
                in normalized_answer
            ):
                matched = True
                matched_value = alternative
                break

        fact_results.append(
            {
                "alternatives":
                    alternatives,

                "matched":
                    matched,

                "matched_value":
                    matched_value,
            }
        )

    matched_count = sum(
        1
        for fact in fact_results
        if fact["matched"]
    )

    factual_coverage = (
        matched_count
        / len(fact_results)
    )

    return (
        factual_coverage,
        fact_results,
    )


# =====================================================
# Main evaluation
# =====================================================

def main():

    # -------------------------------------------------
    # Load evaluation dataset
    # -------------------------------------------------

    data = json.loads(
        EVAL_FILE.read_text(
            encoding="utf-8"
        )
    )

    questions = data[
        "questions"
    ]

    total = len(
        questions
    )

    print(
        "Loading RAG pipeline..."
    )

    rag = LaborLawRAG()

    # -------------------------------------------------
    # Result storage
    # -------------------------------------------------

    results = []

    # -------------------------------------------------
    # Counters
    # -------------------------------------------------

    status_correct_count = 0

    retrieval_correct_count = 0
    retrieval_eval_count = 0

    answer_behavior_correct_count = 0
    answer_count = 0

    clarify_correct_count = 0
    clarify_count = 0

    out_of_scope_correct_count = 0
    out_of_scope_count = 0

    factual_coverage_sum = 0.0
    factual_eval_count = 0

    fully_factual_count = 0

    total_time = 0.0

    # -------------------------------------------------
    # Safety / abstention counters
    # -------------------------------------------------

    abstention_count = 0

    correct_abstention_count = 0
    wrong_abstention_count = 0

    answered_with_wrong_article_count = 0

    # -------------------------------------------------
    # Start
    # -------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        f"Running {total} RAG evaluation questions"
    )

    print(
        "=" * 80
    )

    # =================================================
    # Evaluate each question
    # =================================================

    for index, item in enumerate(
        questions,
        start=1,
    ):
        query = item[
            "query"
        ]

        expected_status = item[
            "expected_status"
        ]

        expected_article = item.get(
            "expected_article"
        )

        expected_facts = item.get(
            "expected_facts",
            [],
        )

        # ---------------------------------------------
        # Run RAG
        # ---------------------------------------------

        start_time = (
            time.perf_counter()
        )

        response = rag.answer(
            query
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        total_time += elapsed

        # ---------------------------------------------
        # Extract response fields
        # ---------------------------------------------

        actual_status = response.get(
            "status",
            "error",
        )

        answer = response.get(
            "answer",
            "",
        )

        clarifying_question = (
            response.get(
                "clarifying_question",
                "",
            )
        )

        sources = response.get(
            "sources",
            [],
        )

        retrieval_score = (
            response.get(
                "retrieval_score",
                None,
            )
        )

        retrieval_k = (
            response.get(
                "retrieval_k",
                None,
            )
        )

        reranker_score = (
            response.get(
                "reranker_score",
                None,
            )
        )

        reranker_confidence = (
            response.get(
                "reranker_confidence",
                None,
            )
        )

        reranker_margin = (
            response.get(
                "reranker_margin",
                None,
            )
        )

        abstained = response.get(
            "abstained",
            False,
        )

        abstention_reason = (
            response.get(
                "abstention_reason",
                None,
            )
        )

        retrieval_candidates = (
            response.get(
                "retrieval_candidates",
                [],
            )
        )

        # =============================================
        # Status accuracy
        # =============================================

        status_correct = (
            actual_status
            == expected_status
        )

        if status_correct:
            status_correct_count += 1

        # =============================================
        # Retrieval accuracy
        # =============================================

        retrieval_hit = None

        if expected_article is not None:
            retrieval_eval_count += 1

            retrieval_hit = (
                expected_article
                in sources
            )

            if retrieval_hit:
                retrieval_correct_count += 1

        # =============================================
        # Behavioral metrics
        # =============================================

        answer_behavior_correct = None
        clarification_correct = None
        out_of_scope_correct = None

        # ---------------------------------------------
        # Expected answer
        # ---------------------------------------------

        if expected_status == "answer":
            answer_count += 1

            answer_behavior_correct = (
                actual_status == "answer"
                and bool(
                    answer.strip()
                )
            )

            if answer_behavior_correct:
                (
                    answer_behavior_correct_count
                ) += 1

        # ---------------------------------------------
        # Expected clarification
        # ---------------------------------------------

        elif expected_status == "clarify":
            clarify_count += 1

            clarification_correct = (
                actual_status == "clarify"
                and bool(
                    clarifying_question.strip()
                )
            )

            if clarification_correct:
                clarify_correct_count += 1

        # ---------------------------------------------
        # Expected out of scope
        # ---------------------------------------------

        elif (
            expected_status
            == "out_of_scope"
        ):
            out_of_scope_count += 1

            out_of_scope_correct = (
                actual_status
                == "out_of_scope"
            )

            if out_of_scope_correct:
                (
                    out_of_scope_correct_count
                ) += 1

        # =============================================
        # Factual coverage
        # =============================================

        factual_coverage = None
        fact_results = []
        fully_factual = None

        if (
            expected_status == "answer"
            and expected_facts
        ):
            factual_eval_count += 1

            if (
                actual_status == "answer"
                and answer.strip()
            ):
                (
                    factual_coverage,
                    fact_results,
                ) = evaluate_expected_facts(
                    answer,
                    expected_facts,
                )

            else:
                factual_coverage = 0.0

                fact_results = [
                    {
                        "alternatives":
                            alternatives,

                        "matched":
                            False,

                        "matched_value":
                            None,
                    }
                    for alternatives
                    in expected_facts
                ]

            factual_coverage_sum += (
                factual_coverage
            )

            fully_factual = (
                factual_coverage
                == 1.0
            )

            if fully_factual:
                fully_factual_count += 1

        # =============================================
        # Abstention analysis
        # =============================================

        if abstained:
            abstention_count += 1

            if (
                expected_status
                == "out_of_scope"
            ):
                correct_abstention_count += 1

            else:
                wrong_abstention_count += 1

        # =============================================
        # Critical safety metric:
        # answered using wrong article
        # =============================================

        answered_with_wrong_article = False

        if (
            expected_status == "answer"
            and actual_status == "answer"
            and expected_article is not None
            and expected_article not in sources
        ):
            answered_with_wrong_article = True

            answered_with_wrong_article_count += 1

        # =============================================
        # Save question result
        # =============================================

        result = {
            "id": item.get(
                "id",
                index,
            ),

            "query":
                query,

            "expected_status":
                expected_status,

            "actual_status":
                actual_status,

            "status_correct":
                status_correct,

            "expected_article":
                expected_article,

            "sources":
                sources,

            "retrieval_hit":
                retrieval_hit,

            "retrieval_k":
                retrieval_k,

            "retrieval_score":
                retrieval_score,

            "reranker_score":
                reranker_score,

            "reranker_confidence":
                reranker_confidence,

            "reranker_margin":
                reranker_margin,

            "retrieval_candidates":
                retrieval_candidates,

            "abstained":
                abstained,

            "abstention_reason":
                abstention_reason,

            "answered_with_wrong_article":
                answered_with_wrong_article,

            "answer_behavior_correct":
                answer_behavior_correct,

            "clarification_correct":
                clarification_correct,

            "out_of_scope_correct":
                out_of_scope_correct,

            "expected_facts":
                expected_facts,

            "factual_coverage":
                factual_coverage,

            "fully_factual":
                fully_factual,

            "fact_results":
                fact_results,

            "answer":
                answer,

            "clarifying_question":
                clarifying_question,

            "latency_s":
                elapsed,
        }

        results.append(
            result
        )

        # =============================================
        # Console output
        # =============================================

        print()
        print(
            f"[{index}/{total}]"
        )

        print(
            f"Q: {query}"
        )

        print(
            f"Expected: "
            f"{expected_status}"
        )

        print(
            f"Actual:   "
            f"{actual_status}"
        )

        print(
            "Status:",
            "✓"
            if status_correct
            else "✗",
        )

        if expected_article:
            print(
                "Expected article:",
                expected_article,
            )

            print(
                "Retrieval:",
                "✓"
                if retrieval_hit
                else "✗",
            )

            if sources:
                print(
                    "Selected source:",
                    ", ".join(
                        sources
                    ),
                )

        if (
            retrieval_score
            is not None
        ):
            print(
                "Dense score: "
                f"{retrieval_score:.4f}"
            )

        if (
            reranker_score
            is not None
        ):
            print(
                "Reranker score: "
                f"{reranker_score:.4f}"
            )

        if (
            reranker_margin
            is not None
        ):
            print(
                "Reranker margin: "
                f"{reranker_margin:.4f}"
            )

        if abstained:
            print(
                "Abstained: "
                f"{abstention_reason}"
            )

        if answered_with_wrong_article:
            print(
                "⚠ CRITICAL: "
                "Answered using wrong article"
            )

        # ---------------------------------------------
        # Answer
        # ---------------------------------------------

        if actual_status == "answer":
            print(
                f"Answer: {answer}"
            )

        # ---------------------------------------------
        # Clarification
        # ---------------------------------------------

        elif (
            actual_status
            == "clarify"
        ):
            print(
                "Clarify:",
                clarifying_question,
            )

        # ---------------------------------------------
        # Factual evaluation
        # ---------------------------------------------

        if (
            factual_coverage
            is not None
        ):
            print(
                "Factual coverage: "
                f"{factual_coverage:.2%}"
            )

            for fact in fact_results:
                symbol = (
                    "✓"
                    if fact["matched"]
                    else "✗"
                )

                print(
                    f"  {symbol} "
                    f"{fact['alternatives']}"
                )

        print(
            f"Latency: "
            f"{elapsed:.2f}s"
        )

    # =================================================
    # Aggregate metrics
    # =================================================

    status_accuracy = (
        status_correct_count
        / total
        if total
        else 0.0
    )

    retrieval_accuracy = (
        retrieval_correct_count
        / retrieval_eval_count
        if retrieval_eval_count
        else 0.0
    )

    answer_behavior_accuracy = (
        answer_behavior_correct_count
        / answer_count
        if answer_count
        else 0.0
    )

    clarification_accuracy = (
        clarify_correct_count
        / clarify_count
        if clarify_count
        else 0.0
    )

    out_of_scope_accuracy = (
        out_of_scope_correct_count
        / out_of_scope_count
        if out_of_scope_count
        else 0.0
    )

    avg_factual_coverage = (
        factual_coverage_sum
        / factual_eval_count
        if factual_eval_count
        else 0.0
    )

    fully_factual_answer_rate = (
        fully_factual_count
        / factual_eval_count
        if factual_eval_count
        else 0.0
    )

    avg_latency = (
        total_time
        / total
        if total
        else 0.0
    )

    abstention_rate = (
        abstention_count
        / total
        if total
        else 0.0
    )

    # =================================================
    # Reranker score analysis
    # =================================================

    in_scope_reranker_scores = []
    out_of_scope_reranker_scores = []

    in_scope_reranker_margins = []
    out_of_scope_reranker_margins = []

    for result in results:

        score = result.get(
            "reranker_score"
        )

        margin = result.get(
            "reranker_margin"
        )

        if (
            result["expected_status"]
            == "out_of_scope"
        ):
            if score is not None:
                out_of_scope_reranker_scores.append(
                    score
                )

            if margin is not None:
                out_of_scope_reranker_margins.append(
                    margin
                )

        else:
            if score is not None:
                in_scope_reranker_scores.append(
                    score
                )

            if margin is not None:
                in_scope_reranker_margins.append(
                    margin
                )

    # -------------------------------------------------
    # Score statistics
    # -------------------------------------------------

    avg_in_scope_reranker_score = (
        sum(in_scope_reranker_scores)
        / len(in_scope_reranker_scores)
        if in_scope_reranker_scores
        else None
    )

    min_in_scope_reranker_score = (
        min(in_scope_reranker_scores)
        if in_scope_reranker_scores
        else None
    )

    avg_out_of_scope_reranker_score = (
        sum(out_of_scope_reranker_scores)
        / len(out_of_scope_reranker_scores)
        if out_of_scope_reranker_scores
        else None
    )

    max_out_of_scope_reranker_score = (
        max(out_of_scope_reranker_scores)
        if out_of_scope_reranker_scores
        else None
    )

    # -------------------------------------------------
    # Margin statistics
    # -------------------------------------------------

    avg_in_scope_reranker_margin = (
        sum(in_scope_reranker_margins)
        / len(in_scope_reranker_margins)
        if in_scope_reranker_margins
        else None
    )

    min_in_scope_reranker_margin = (
        min(in_scope_reranker_margins)
        if in_scope_reranker_margins
        else None
    )

    avg_out_of_scope_reranker_margin = (
        sum(out_of_scope_reranker_margins)
        / len(out_of_scope_reranker_margins)
        if out_of_scope_reranker_margins
        else None
    )

    max_out_of_scope_reranker_margin = (
        max(out_of_scope_reranker_margins)
        if out_of_scope_reranker_margins
        else None
    )

    # =================================================
    # Summary
    # =================================================

    summary = {
        "total_questions":
            total,

        "status_accuracy":
            status_accuracy,

        "retrieval_accuracy":
            retrieval_accuracy,

        "answer_behavior_accuracy":
            answer_behavior_accuracy,

        "clarification_accuracy":
            clarification_accuracy,

        "out_of_scope_accuracy":
            out_of_scope_accuracy,

        "avg_factual_coverage":
            avg_factual_coverage,

        "fully_factual_answer_rate":
            fully_factual_answer_rate,

        "avg_latency_s":
            avg_latency,

        "abstention_count":
            abstention_count,

        "abstention_rate":
            abstention_rate,

        "correct_abstention_count":
            correct_abstention_count,

        "wrong_abstention_count":
            wrong_abstention_count,

        "answered_with_wrong_article_count":
            answered_with_wrong_article_count,

        "avg_in_scope_reranker_score":
            avg_in_scope_reranker_score,

        "min_in_scope_reranker_score":
            min_in_scope_reranker_score,

        "avg_out_of_scope_reranker_score":
            avg_out_of_scope_reranker_score,

        "max_out_of_scope_reranker_score":
            max_out_of_scope_reranker_score,

        "avg_in_scope_reranker_margin":
            avg_in_scope_reranker_margin,

        "min_in_scope_reranker_margin":
            min_in_scope_reranker_margin,

        "avg_out_of_scope_reranker_margin":
            avg_out_of_scope_reranker_margin,

        "max_out_of_scope_reranker_margin":
            max_out_of_scope_reranker_margin,

        "embedding_model":
            "BAAI/bge-m3",

        "reranker_model":
            "BAAI/bge-reranker-v2-m3",

        "generator":
            "Gemma 3 4B Q4_K_M",

        "retrieval_top_k":
            3,

        "selected_context_articles":
            1,
    }

    # =================================================
    # Save JSON
    # =================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            {
                "summary":
                    summary,

                "results":
                    results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =================================================
    # Final summary
    # =================================================

    print()
    print(
        "=" * 80
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        "Status Accuracy: "
        f"{status_accuracy:.2%}"
    )

    print(
        "Retrieval Accuracy: "
        f"{retrieval_accuracy:.2%}"
    )

    print(
        "Answer Behavior Accuracy: "
        f"{answer_behavior_accuracy:.2%}"
    )

    print(
        "Clarification Accuracy: "
        f"{clarification_accuracy:.2%}"
    )

    print(
        "Out-of-Scope Accuracy: "
        f"{out_of_scope_accuracy:.2%}"
    )

    print(
        "Average Factual Coverage: "
        f"{avg_factual_coverage:.2%}"
    )

    print(
        "Fully Factual Answer Rate: "
        f"{fully_factual_answer_rate:.2%}"
    )

    print(
        "Average Latency: "
        f"{avg_latency:.2f}s"
    )

    print()

    print(
        "SAFETY METRICS"
    )

    print(
        "Abstentions: "
        f"{abstention_count}/{total} "
        f"({abstention_rate:.2%})"
    )

    print(
        "Correct OOD Abstentions: "
        f"{correct_abstention_count}"
    )

    print(
        "In-Scope Abstentions: "
        f"{wrong_abstention_count}"
    )

    print(
        "Answered With Wrong Article: "
        f"{answered_with_wrong_article_count}"
    )

    print()

    print(
        "RERANKER SCORE ANALYSIS"
    )

    if (
        avg_in_scope_reranker_score
        is not None
    ):
        print(
            "Average In-Scope Reranker Score: "
            f"{avg_in_scope_reranker_score:.4f}"
        )

    if (
        min_in_scope_reranker_score
        is not None
    ):
        print(
            "Minimum In-Scope Reranker Score: "
            f"{min_in_scope_reranker_score:.4f}"
        )

    if (
        avg_out_of_scope_reranker_score
        is not None
    ):
        print(
            "Average OOD Reranker Score: "
            f"{avg_out_of_scope_reranker_score:.4f}"
        )

    if (
        max_out_of_scope_reranker_score
        is not None
    ):
        print(
            "Maximum OOD Reranker Score: "
            f"{max_out_of_scope_reranker_score:.4f}"
        )

    print()

    print(
        "RERANKER MARGIN ANALYSIS"
    )

    if (
        avg_in_scope_reranker_margin
        is not None
    ):
        print(
            "Average In-Scope Margin: "
            f"{avg_in_scope_reranker_margin:.4f}"
        )

    if (
        min_in_scope_reranker_margin
        is not None
    ):
        print(
            "Minimum In-Scope Margin: "
            f"{min_in_scope_reranker_margin:.4f}"
        )

    if (
        avg_out_of_scope_reranker_margin
        is not None
    ):
        print(
            "Average OOD Margin: "
            f"{avg_out_of_scope_reranker_margin:.4f}"
        )

    if (
        max_out_of_scope_reranker_margin
        is not None
    ):
        print(
            "Maximum OOD Margin: "
            f"{max_out_of_scope_reranker_margin:.4f}"
        )

    print()

    print(
        f"✓ Results saved to "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()