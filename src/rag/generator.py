import os
import requests


BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "http://127.0.0.1:8081"
)


class QwenGenerator:
    def __init__(self):
        print("Using local Gemma 3 4B Q4_K_M via llama.cpp")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:

        payload = {
            "model": "gemma-3-4b-it",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.2,
            "max_tokens": max_new_tokens,
        }

        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        message = data["choices"][0]["message"]

        answer = message.get(
            "content",
            ""
        ).strip()

        return answer