import os
import logging

logger = logging.getLogger(__name__)


class GroqClient:
    def __init__(self, model="llama-3.1-8b-instant"):
        from groq import Groq
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model
        logger.info("GroqClient initialized with model: %s", self.model)

    def __call__(self, messages, max_new_tokens=2048, **kwargs):
        logger.debug("GroqClient: sending %d message(s), max_new_tokens=%d", len(messages), max_new_tokens)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens
        )
        finish_reason = response.choices[0].finish_reason
        logger.debug("GroqClient: received response, finish_reason=%s", finish_reason)
        return [{"generated_text": messages + [{"role": "assistant", "content": response.choices[0].message.content}]}]