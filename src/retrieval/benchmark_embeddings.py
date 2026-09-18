import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DOCUMENTS_FILE = Path("data/processed/labor_law_documents.json")
EVAL_FILE = Path("evaluation/retrieval_eval.json")

MODELS = {
    "bge": {
        "name": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "e5": {
        "name": "intfloat/multilingual-e5-large",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def best_reciprocal_rank(results, expected_articles):
    for rank, heading in enumerate(results, start=1):
        if heading in expected_articles:
            return 1 / rank
    return 0.0


def main():
    if len(sys.argv) != 2 or sys.argv[1].lower() not in MODELS:
        print("Usage:")
        print("  python src/retrieval/benchmark_embeddings.py bge")
        print("  python src/retrieval/benchmark_embeddings.py e5")
        sys.exit(1)

    model_key = sys.argv[1].lower()
    config = MODELS[model_key]

    documents = load_json(DOCUMENTS_FILE)
    eval_data = load_json(EVAL_FILE)
    questions = eval_data["questions"]

    print("=" * 72)
    print(f"MODEL: {model_key} | {config['name']}")
    print(f"Documents: {len(documents)} | Evaluation questions: {len(questions)}")
    print("=" * 72)

    t0 = time.perf_counter()
    model = SentenceTransformer(config["name"])
    model_load_time = time.perf_counter() - t0

    passages = [
        config["passage_prefix"] + doc["text_normalized"]
        for doc in documents
    ]

    t0 = time.perf_counter()
    embeddings = model.encode(
        passages,
        batch_size=4,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    corpus_embedding_time = time.perf_counter() - t0

    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    hits_1 = hits_3 = hits_5 = 0
    reciprocal_ranks = []
    query_times = []
    details = []

    for item in questions:
        query_text = config["query_prefix"] + item["query"]

        t0 = time.perf_counter()
        q_emb = model.encode([query_text], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        scores, indices = index.search(q_emb, 5)
        query_times.append(time.perf_counter() - t0)

        headings = [documents[i]["article_heading"] for i in indices[0]]
        expected = item["expected_articles"]

        hit1 = any(x in expected for x in headings[:1])
        hit3 = any(x in expected for x in headings[:3])
        hit5 = any(x in expected for x in headings[:5])

        hits_1 += int(hit1)
        hits_3 += int(hit3)
        hits_5 += int(hit5)
        rr = best_reciprocal_rank(headings, expected)
        reciprocal_ranks.append(rr)

        details.append({
            "id": item["id"],
            "category": item["category"],
            "query": item["query"],
            "expected_articles": expected,
            "top5": [
                {"rank": rank, "article": heading, "score": float(score)}
                for rank, (heading, score) in enumerate(zip(headings, scores[0]), start=1)
            ],
            "hit@1": hit1,
            "hit@3": hit3,
            "hit@5": hit5,
            "reciprocal_rank": rr,
        })

        marker = "✓" if hit1 else ("~" if hit3 else "✗")
        print(f"{marker} Q{item['id']:02d}: {item['query']}")
        print(f"   Expected: {', '.join(expected)}")
        print(f"   Top-3: {', '.join(headings[:3])}")

    n = len(questions)
    summary = {
        "model": model_key,
        "model_name": config["name"],
        "evaluation_questions": n,
        "recall@1": hits_1 / n,
        "recall@3": hits_3 / n,
        "recall@5": hits_5 / n,
        "mrr": sum(reciprocal_ranks) / n,
        "avg_query_time_ms": sum(query_times) / len(query_times) * 1000,
        "corpus_embedding_time_s": corpus_embedding_time,
        "model_load_time_s": model_load_time,
        "dimension": int(embeddings.shape[1]),
    }

    output = {
        "summary": summary,
        "details": details,
    }

    out_path = Path(f"benchmarks/embedding_{model_key}_45q.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\n✓ Saved: {out_path}")


if __name__ == "__main__":
    main()
