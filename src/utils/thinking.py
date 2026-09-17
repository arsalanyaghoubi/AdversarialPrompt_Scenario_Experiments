import re

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

    DeepSeek R1 format — two variants:
        Full:    <think>...reasoning...</think>...answer...
        Partial: ...reasoning...</think>...answer...
        (The opening <think> token is stripped by skip_special_tokens=True
         but </think> survives as a regular text token.)

    GPT-OSS format:
        <|channel|>analysis<|message|>...reasoning...<|channel|>response<|message|>...answer...
    """
    # --- DeepSeek R1 ---
    # Split on </think> rather than matching both tags so the partial case
    # (opening tag stripped) is handled identically to the full case.
    if '</think>' in raw_text:
        parts = raw_text.split('</think>', 1)
        # Remove the opening <think> tag if it survived, then strip whitespace.
        thinking = parts[0].replace('<think>', '').strip()
        answer   = parts[1].strip()
        return {"thinking": thinking or None, "answer": answer}

    # --- GPT-OSS ---
    match = _GPT_OSS_THINKING_RE.search(raw_text)
    if match:
        thinking = match.group(1).strip()
        answer   = _GPT_OSS_THINKING_RE.sub("", raw_text)
        answer   = _GPT_OSS_CHANNEL_RE.sub("", answer).strip()
        return {"thinking": thinking or None, "answer": answer}

    # No thinking block found (non-reasoning models)
    return {"thinking": None, "answer": raw_text.strip()}