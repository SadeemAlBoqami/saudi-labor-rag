from typing import List, Dict


def build_context(results: List[Dict], top_k: int = 3) -> str:
    """
    Convert retrieved legal articles into a structured context
    that can later be sent to the LLM.
    """

    selected = results[:top_k]

    blocks = []

    for item in selected:
        block = (
            f"[{item['article_heading']}]\n"
            f"{item['text_verbatim']}"
        )

        blocks.append(block)

    return "\n\n---\n\n".join(blocks)
