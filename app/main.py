from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.rag.rag_pipeline import LaborLawRAG


app = FastAPI(
    title="Saudi Labor Law RAG",
    version="0.1.0",
)

rag = LaborLawRAG()


class QuestionRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "saudi-labor-rag",
    }


@app.post("/api/ask")
def ask_question(request: QuestionRequest):
    result = rag.answer(request.query)

    return {
        "status": result.get("status"),
        "answer": result.get("answer", ""),
        "clarifying_question": result.get(
            "clarifying_question",
            "",
        ),
        "sources": result.get("sources", []),
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>مساعد نظام العمل السعودي</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f5f7fa;
            margin: 0;
            padding: 40px 20px;
        }

        .container {
            max-width: 800px;
            margin: auto;
            background: white;
            padding: 32px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }

        h1 {
            margin-top: 0;
        }

        textarea {
            width: 100%;
            min-height: 100px;
            padding: 14px;
            font-size: 16px;
            box-sizing: border-box;
        }

        button {
            margin-top: 12px;
            padding: 12px 24px;
            font-size: 16px;
            cursor: pointer;
        }

        .result {
            margin-top: 28px;
            padding: 20px;
            background: #f8f9fb;
            border-radius: 12px;
            display: none;
        }

        .sources {
            margin-top: 18px;
        }

        .status {
            font-size: 14px;
            opacity: 0.7;
        }
    </style>
</head>

<body>

<div class="container">

    <h1>مساعد نظام العمل السعودي</h1>

    <p>
        اسأل عن أحكام نظام العمل السعودي،
        وسيجيب النظام اعتمادًا على النصوص النظامية المسترجعة.
    </p>

    <textarea
        id="query"
        placeholder="مثال: كم مدة فترة التجربة؟"
    ></textarea>

    <button onclick="askQuestion()">
        اسأل
    </button>

    <div id="result" class="result">

        <div id="status" class="status"></div>

        <h3>الإجابة</h3>
        <div id="answer"></div>

        <div id="clarification"></div>

        <div class="sources">
            <h3>الأساس النظامي</h3>
            <div id="sources"></div>
        </div>

    </div>

</div>


<script>

async function askQuestion() {

    const query =
        document.getElementById("query").value.trim();

    if (!query) {
        return;
    }

    const button =
        document.querySelector("button");

    button.disabled = true;
    button.textContent = "جاري البحث...";

    try {

        const response = await fetch(
            "/api/ask",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    query: query
                })
            }
        );

        const data = await response.json();

        document.getElementById("result").style.display =
            "block";

        document.getElementById("status").textContent =
            "الحالة: " + data.status;

        if (data.status === "clarify") {

            document.getElementById("answer").textContent =
                "";

            document.getElementById(
                "clarification"
            ).textContent =
                data.clarifying_question;

        } else {

            document.getElementById("answer").textContent =
                data.answer || "";

            document.getElementById(
                "clarification"
            ).textContent =
                "";
        }

        document.getElementById("sources").textContent =
            data.sources && data.sources.length
                ? data.sources.join("، ")
                : "لا توجد مصادر معروضة.";

    } catch (error) {

        document.getElementById("result").style.display =
            "block";

        document.getElementById("answer").textContent =
            "حدث خطأ أثناء الاتصال بالنظام.";

    } finally {

        button.disabled = false;
        button.textContent = "اسأل";
    }
}

</script>

</body>
</html>
"""
