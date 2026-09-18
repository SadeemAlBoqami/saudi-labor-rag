import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen3-4B"


class QwenGenerator:
    def __init__(self):
        print("Loading Qwen3-4B tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        print("Loading Qwen3-4B model...")

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )

        print("✓ Qwen3-4B loaded")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
            )

        generated_tokens = outputs[
            0,
            inputs["input_ids"].shape[-1]:
        ]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return answer.strip()
