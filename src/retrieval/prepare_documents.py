import json
import re
from pathlib import Path

INPUT_FILE = Path("data/processed/labor_law_structured.json")
OUTPUT_FILE = Path("data/processed/labor_law_documents.json")


def normalize_arabic(text: str) -> str:
    text = text.strip()

    # remove tatweel
    text = text.replace("ـ", "")

    # normalize alef variants for retrieval only
    text = re.sub(r"[إأآا]", "ا", text)

    # normalize ya
    text = text.replace("ى", "ي")

    # normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    articles = data["document"]["articles"]

    documents = []

    for article in articles:
        heading = article["article_heading"].strip()
        text_verbatim = article["article_text"].strip()
        footnotes = article.get("footnotes", [])

        retrieval_text = f"{heading}\n{text_verbatim}"

        documents.append(
            {
                "source_id": "labor_law",
                "document_title": data["document"]["title"],
                "article_heading": heading,
                "text_verbatim": text_verbatim,
                "text_normalized": normalize_arabic(retrieval_text),
                "footnotes": footnotes,
            }
        )

    OUTPUT_FILE.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✓ Documents created: {len(documents)}")
    print(f"✓ Saved to: {OUTPUT_FILE}")

    print("\nSample:")
    print(json.dumps(documents[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()