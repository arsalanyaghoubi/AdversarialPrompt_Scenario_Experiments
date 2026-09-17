import pathlib
import json
import re
import logging
from utils.thinking import parse_thinking_output
from utils.config import MIN_WORD_COUNT, MAX_WORD_COUNT

logger = logging.getLogger(__name__)

BASE_DIR = pathlib.Path(__file__).parent.parent.parent
CONTEXT_DIR = BASE_DIR / "Context"

with open(BASE_DIR / "config.json") as f:
    CONFIG = json.load(f)

CF_TARGET     = CONFIG["targets"]["CF"]
SUM_TARGET    = CONFIG["targets"]["SUM"]
PAR_TARGET    = CONFIG["targets"]["PAR"]
SUM_PAR_TARGET = CONFIG["targets"]["SUM_PAR"]
N_PARAGRAPHS  = CONFIG["n_paragraphs"]

SCENARIOS = [
    {
        "dir": "1_OrigCF_Context",
        "scenario_id": "ConsentForm",
        "context_type": "consent form",
        "needs_cf": True,
        "needs_summary": False,
        "needs_paragraph": False,
        "needs_sum_par": False,
        "prompt_prefix": "CF_Adversarial_Prompt",
        "result_prefix": "OrigCF_Context",
        "target": CF_TARGET,
    },
    {
        "dir": "2_SumCF_Context",
        "scenario_id": "CF_SUM",
        "context_type": "summary",
        "needs_cf": False,
        "needs_summary": True,
        "needs_paragraph": False,
        "needs_sum_par": False,
        "prompt_prefix": "CF_SUM_Adversarial_Prompt",
        "result_prefix": "SumCF_Context",
        "target": SUM_TARGET,
    },
    {
        "dir": "3_SUM_PARAG_Context",
        "scenario_id": "SUM & PAR",
        "context_type": "summary and paragraph",
        "needs_cf": False,
        "needs_summary": False,
        "needs_paragraph": False,
        "needs_sum_par": True,
        "prompt_prefix": "SUM_PAR_Adversarial_Prompt",
        "result_prefix": "SUM_PARAG_Context",
        "target": SUM_PAR_TARGET,
    },
    {
        "dir": "4_PARAG_Context",
        "scenario_id": "PAR",
        "context_type": "paragraph",
        "needs_cf": False,
        "needs_summary": False,
        "needs_paragraph": True,
        "needs_sum_par": False,
        "prompt_prefix": "PARAG_Adversarial_Prompt",
        "result_prefix": "PARAG_Context",
        "target": PAR_TARGET,
    },
]

CRITERIA = ["C1", "C2", "C3"]

# Prompts that are clearly invalid model outputs (placeholder text, criterion names, etc.)
INVALID_PROMPTS = {"C1", "C2", "C3", "...", "{generatedadversarialprompt}", "generated prompt"}


# ── Shared helpers ─────────────────────────────────────────────────────────────

def format_prompt(prompt):
    """Extract plain prompt text from whatever shape the model returned."""
    if isinstance(prompt, dict):
        return prompt.get("adversarial_prompt", prompt.get("prompt", str(prompt)))
    return str(prompt)


def validate_word_count(sentence: str) -> str | None:
    """Return sentence if it is within the configured word-count window, else None."""
    if not isinstance(sentence, str):
        return None
    word_count = len(sentence.split())
    if MIN_WORD_COUNT <= word_count <= MAX_WORD_COUNT:
        return sentence
    return None


def _extract_json_objects(text: str) -> list:
    """
    Extract JSON objects from text using brace-depth tracking.
    More robust than regex: handles nested structures and list values
    (e.g. "criteria_targeted": ["C1", "C2"]).
    """
    objects = []
    depth = 0
    start = -1
    in_string = False
    escape_next = False
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    obj = json.loads(text[start:i + 1])
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start = -1
    return objects


# ── Prompt loading ─────────────────────────────────────────────────────────────

def load_system_prompt(scenario, criterion):
    prompt_file = BASE_DIR / scenario["dir"] / f"{scenario['prompt_prefix']}_{criterion}.txt"
    if not prompt_file.exists():
        logger.warning("Prompt file %s does not exist.", prompt_file)
        return None
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    return content.replace("{n_prompts}", str(scenario["target"]))


def build_user_message(scenario, context_content, cf_filename):
    """Build the user-side message from the scenario type and context."""
    if scenario["needs_cf"]:
        return f"Consent Form File: {cf_filename}\n\nConsent Form:\n{context_content}"
    elif scenario["needs_sum_par"]:
        return f"Consent Form File: {cf_filename}\n\nSummary and Paragraph:\n{context_content}"
    elif scenario["needs_summary"]:
        return f"Consent Form File: {cf_filename}\n\nSummary:\n{context_content}"
    elif scenario["needs_paragraph"]:
        return f"Consent Form File: {cf_filename}\n\nParagraph:\n{context_content}"
    else:
        return "Generate the adversarial prompts without context."


# ── Core generation ────────────────────────────────────────────────────────────

def generate_batch(scenario, criterion, context_content, cf_filename, client,
                   generated_so_far=None):
    """
    Call the model once and return a list of validated prompt dicts.

    Uses getattr(client, 'MAX_NEW_TOKENS', 2048) so ThinkingClient (32768)
    and hf_pipeline (2048) are handled without a circular import.
    """
    system_prompt = load_system_prompt(scenario, criterion)
    if system_prompt is None:
        return []

    user_message = build_user_message(scenario, context_content, cf_filename)
    final_user_message = (
        f"Use the following {scenario['context_type']} as the context to generate adversarial prompts:\n"
        + user_message
    )
    if generated_so_far:
        prompt_list = "\n".join(f'"{p}"' for p in generated_so_far)
        final_user_message += f"\n\nDo NOT generate any of these already-generated prompts:\n{prompt_list}"

    # ThinkingClient exposes MAX_NEW_TOKENS = 32768; hf_pipeline falls back to 2048.
    max_new_tokens = getattr(client, 'MAX_NEW_TOKENS', 2048)

    try:
        response = client(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": final_user_message}],
            max_new_tokens=max_new_tokens,
        )
    except Exception as e:
        if "roles must alternate" in str(e):
            logger.warning(
                "Model %s does not support system role — merging into user message.",
                type(client).__name__,
            )
            response = client(
                [{"role": "user", "content": f"{system_prompt}\n\n{final_user_message}"}],
                max_new_tokens=max_new_tokens,
            )
        else:
            raise

    raw_content = response[0]["generated_text"][-1]["content"].strip()
    parsed_output = parse_thinking_output(raw_content)
    result_text = parsed_output["answer"]
    # batch_reasoning is the chain-of-thought block; None for non-reasoning models.
    batch_reasoning = parsed_output["thinking"]

    logger.info("Raw output before parsing:\n%s", result_text)

    results = []

    # Parser 1: JSON array
    s, e = result_text.find('['), result_text.rfind(']')
    if s != -1 and e != -1 and '{' in result_text[s:e]:
        try:
            results = json.loads(result_text[s:e + 1])
            if results:
                logger.info("Parser 1 (JSON array): %d items", len(results))
        except json.JSONDecodeError:
            pass

    # Parser 2: individual JSON objects via brace-depth tracking
    if not results:
        results = _extract_json_objects(result_text)
        if results:
            logger.info("Parser 2 (JSON objects): %d items", len(results))

    # Parser 3: prose fallback — quoted strings of 20-500 chars
    if not results:
        for match in re.finditer(r'"([^"\n]{20,500})"', result_text):
            matched_text = match.group(1).strip()
            if '**' not in matched_text and not matched_text.startswith((']', '-', '\n')):
                results.append({"prompt": matched_text})
        if results:
            logger.info("Parser 3 (prose fallback): %d items", len(results))

    # Parser 4: markdown list fallback
    if not results:
        for match in re.finditer(r'\*\*Prompt \d+.*?\*\*\s*\n\s*"([^"]{20,})"',
                                  result_text, re.DOTALL):
            results.append({"prompt": match.group(1).strip()})
        if results:
            logger.info("Parser 4 (markdown fallback): %d items", len(results))

    if not results:
        logger.warning("All parsers failed. Raw output (first 500 chars): %s", result_text[:500])

    # Normalise — some models return stringified JSON objects inside an array.
    normalized = []
    for r in results:
        if isinstance(r, str):
            stripped = r.strip()
            if stripped.startswith('{') and stripped.endswith('}'):
                try:
                    r = json.loads(stripped)
                except json.JSONDecodeError:
                    r = {"prompt": stripped}
            else:
                r = {"prompt": stripped}
        if isinstance(r, dict):
            if batch_reasoning:
                r["_reasoning"] = batch_reasoning
            normalized.append(r)
        else:
            logger.warning("Unexpected result type %s: %s", type(r).__name__, str(r)[:100])

    filtered = [
        r for r in normalized
        if format_prompt(r) not in INVALID_PROMPTS
           and validate_word_count(format_prompt(r)) is not None
    ]

    if len(filtered) < len(normalized):
        logger.warning(
            "Filtered %d/%d items (word count or invalid): bounds=[%d, %d]",
            len(normalized) - len(filtered), len(normalized),
            MIN_WORD_COUNT, MAX_WORD_COUNT,
        )

    return filtered


def accumulate_prompts(scenario, criterion, context_content, cf_filename, client,
                       generated_so_far=None):
    """
    Retry generate_batch until scenario["target"] unique prompts are collected
    or the safety caps are hit.

    generated_so_far: a set of already-generated prompt strings shared across
    scenarios for deduplication. Passed in from _run_for_model.
    """
    accumulated = []
    retries = 0
    max_iterations = scenario["target"] * 5
    iterations = 0

    while len(accumulated) < scenario["target"] and iterations < max_iterations:
        iterations += 1
        if retries >= 3:
            logger.warning(
                "Max retries reached for %s %s, skipping.",
                scenario["scenario_id"], criterion,
            )
            break

        new_prompts = generate_batch(
            scenario, criterion, context_content, cf_filename, client,
            generated_so_far=generated_so_far,
        )
        unique_prompts = [
            p for p in new_prompts
            if generated_so_far is None or format_prompt(p) not in generated_so_far
        ]
        if not unique_prompts:
            retries += 1
            logger.warning("Retry %d/3 for %s %s.", retries, scenario["scenario_id"], criterion)
        else:
            retries = 0
            accumulated.extend(unique_prompts)

    if iterations >= max_iterations:
        logger.warning(
            "Max iterations (%d) reached for %s %s with %d/%d prompts collected.",
            max_iterations, scenario["scenario_id"], criterion,
            len(accumulated), scenario["target"],
        )
    logger.info(
        "Accumulated %d/%d prompts for %s %s.",
        len(accumulated), scenario["target"], scenario["scenario_id"], criterion,
    )
    return accumulated[:scenario["target"]]