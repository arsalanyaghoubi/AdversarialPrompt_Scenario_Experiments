import re

# DeepSeek R1: <think>...</think>
_DEEPSEEK_RE = re.compile(r"(?s)<think>(.*?)</think>")

# GPT-OSS actual format observed from debug output:
#   <|channel|>analysis<|message|>...thinking...<|channel|>response<|message|>...answer...
#
# Lookahead (?=<\|channel\|>|\Z) stops at the next channel marker without
# consuming it, so _GPT_OSS_CHANNEL_RE can still strip it from the answer.
_GPT_OSS_THINKING_RE = re.compile(
    r"(?s)<\|channel\|>analysis<\|message\|>(.*?)(?=<\|channel\|>|\Z)"
)
_GPT_OSS_CHANNEL_RE = re.compile(r"<\|channel\|>\w+<\|message\|>\n?")


def parse_thinking_output(raw_text: str) -> dict:
    """
    Parses raw output from DeepSeek R1 or GPT-OSS.
    Returns {"thinking": str | None, "answer": str}.

    DeepSeek R1 format:
        <think>...reasoning...</think>
        ...answer...

    GPT-OSS format:
        <|channel|>analysis<|message|>
        ...reasoning...
        <|channel|>response<|message|>
        ...answer...
    """
    # --- DeepSeek R1 ---
    match = _DEEPSEEK_RE.search(raw_text)
    if match:
        thinking = match.group(1).strip()
        answer = _DEEPSEEK_RE.sub("", raw_text).strip()
        return {"thinking": thinking or None, "answer": answer}

    # --- GPT-OSS ---
    match = _GPT_OSS_THINKING_RE.search(raw_text)
    if match:
        thinking = match.group(1).strip()
        # Remove the entire analysis block, then strip remaining channel headers
        answer = _GPT_OSS_THINKING_RE.sub("", raw_text)
        answer = _GPT_OSS_CHANNEL_RE.sub("", answer).strip()
        return {"thinking": thinking or None, "answer": answer}

    # No thinking block found (non-reasoning models)
    return {"thinking": None, "answer": raw_text.strip()}