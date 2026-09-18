from src.rag.rag_pipeline import LaborLawRAG


rag = LaborLawRAG()

questions = [
    "كم مدة فترة التجربة؟",
    "هل أستحق مكافأة نهاية الخدمة؟",
    "كم نسبة ضريبة القيمة المضافة؟",
]


for query in questions:
    result = rag.answer(query)

    print()
    print("=" * 80)
    print("QUESTION:")
    print(query)

    print()
    print("STATUS:")
    print(result["status"])

    if result["status"] == "answer":
        print()
        print("ANSWER:")
        print(result["answer"])

    elif result["status"] == "clarify":
        print()
        print("CLARIFY:")
        print(
            result["clarifying_question"]
        )

    elif result["status"] == "out_of_scope":
        print()
        print("ANSWER:")
        print(result["answer"])

    if result["sources"]:
        print()
        print("SOURCES:")

        for source in result["sources"]:
            print(f"- {source}")