import requests


API_URL = "http://127.0.0.1:8080/v1/chat/completions"


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
            API_URL,
            json=payload,
            timeout=300,
        )

        response.raise_for_status()

        data = response.json()

        message = data["choices"][0]["message"]

        answer = message.get(
            "content",
            ""
        ).strip()

        return answer