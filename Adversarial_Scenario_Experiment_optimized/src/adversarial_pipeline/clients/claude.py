import os
import logging

logger = logging.getLogger(__name__)


class ClaudeClient:
    def __init__(self, model="claude-sonnet-4-6"):
        from anthropic import Anthropic
        self.client = Anthropic()
        self.model = model
        logger.info("ClaudeClient initialized with model: %s", self.model)

    def __call__(self, messages, max_new_tokens=2048, **kwargs):
        system = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)
        logger.debug("ClaudeClient: sending %d user message(s), max_new_tokens=%d", len(user_messages), max_new_tokens)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_new_tokens,
            system=system,
            messages=user_messages
        )
        logger.debug("ClaudeClient: received response, stop_reason=%s", response.stop_reason)
        return [{"generated_text": messages + [{"role": "assistant", "content": response.content[0].text}]}]