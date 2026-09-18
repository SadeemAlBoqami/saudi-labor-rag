import json
from pathlib import Path
from collections import Counter

FILE = Path("data/processed/labor_law_structured_final.json")

def main():
    data = json.loads(FILE.read_text(encoding="utf-8"))

    articles = data["document"]["articles"]

    print("=" * 60)
    print("SAUDI LABOR LAW — CORPUS VALIDATION")
    print("=" * 60)

    print(f"Articles: {len(articles)}")

    empty_text = []
    missing_heading = []
    unclear = []
    repeated_headings = []

    headings = [
        a.get("article_heading", "").strip()
        for a in articles
    ]

    counts = Counter(headings)

    for heading, count in counts.items():
        if heading and count > 1:
            repeated_headings.append((heading, count))

    for i, article in enumerate(articles, start=1):

        heading = article.get("article_heading", "").strip()
        text = article.get("article_text", "").strip()

        if not heading:
            missing_heading.append(i)

        if not text:
            empty_text.append(heading or f"index={i}")

        combined = json.dumps(
            article,
            ensure_ascii=False
        )

        if "[UNCLEAR]" in combined:
            unclear.append(heading or f"index={i}")

    print(f"Empty article texts: {len(empty_text)}")
    print(f"Missing headings: {len(missing_heading)}")
    print(f"Duplicate headings: {len(repeated_headings)}")
    print(f"Articles containing [UNCLEAR]: {len(unclear)}")

    print("\nFirst 5 articles:")
    for article in articles[:5]:
        print(" •", article.get("article_heading"))

    print("\nLast 5 articles:")
    for article in articles[-5:]:
        print(" •", article.get("article_heading"))

    if repeated_headings:
        print("\nDuplicate headings:")
        for heading, count in repeated_headings:
            print(f" • {heading} × {count}")

    if empty_text:
        print("\nEmpty articles:")
        for item in empty_text:
            print(" •", item)

    if unclear:
        print("\n[UNCLEAR] found in:")
        for item in unclear:
            print(" •", item)

    print("\n" + "=" * 60)

    if (
        not empty_text
        and not missing_heading
        and not repeated_headings
        and not unclear
    ):
        print("✓ BASIC STRUCTURAL VALIDATION PASSED")
    else:
        print("⚠ REVIEW REQUIRED")

    print("=" * 60)


if __name__ == "__main__":
    main()
