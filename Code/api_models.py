

class ClaudeClient:
    def __init__(self, model="claude-sonnet-4-6"):
        from anthropic import Anthropic
        self.client = Anthropic()
        self.model = model

    def __call__(self, messages, max_new_tokens=2048, **kwargs):
        system = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_new_tokens,
            system=system,
            messages=user_messages
        )
        return [{"generated_text": messages + [{"role": "assistant", "content": response.content[0].text}]}]

class GroqClient:
    def __init__(self, model="llama-3.1-8b-instant"):
        from groq import Groq
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model

    def __call__(self, messages, max_new_tokens=2048, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens
        )
        return [{"generated_text": messages + [{"role": "assistant", "content": response.choices[0].message.content}]}]
