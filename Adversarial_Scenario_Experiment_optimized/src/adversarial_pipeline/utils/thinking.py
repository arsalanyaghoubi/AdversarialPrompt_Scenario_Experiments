import re
 
# DeepSeek R1: <think>...</think>
_DEEPSEEK_RE = re.compile(r"(?s)<think>(.*?)</think>")
 
# GPT-OSS: analysis block runs from <|message|> through <|end|>
_GPT_OSS_THINKING_RE = re.compile(
    r"(?s)<\|start\|>assistant channel=analysis<\|message\|>(.*?)(?:<\|end\|>|\Z)"
)
_GPT_OSS_CHANNEL_RE = re.compile(r"<\|start\|>assistant channel=\w+<\|message\|>\n?")
 
 
def parse_thinking_output(raw_text: str) -> dict:
    """
    Parses raw output from DeepSeek R1 or GPT-OSS.
    Returns {"thinking": str | None, "answer": str}.
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
        answer = _GPT_OSS_THINKING_RE.sub("", raw_text)
        answer = _GPT_OSS_CHANNEL_RE.sub("", answer).strip()
        return {"thinking": thinking or None, "answer": answer}
 
    # No thinking block found
    return {"thinking": None, "answer": raw_text.strip()}
