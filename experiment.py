import json
import logging
import pathlib
import re
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline as hf_pipeline
from pipeline_code.preprocess import fix_bold_headings, generate_paragraph, generate_summary, save_file
from pipeline_code.prompt_gen import build_user_message, CRITERIA, SCENARIOS
from utils.log_config import setup_logging
from utils.thinking import parse_thinking_output
from utils.config import (
    MIN_WORD_COUNT, MAX_WORD_COUNT, REPETITION_PENALTY,
    AP_MODEL_PATHS, SUM_MODEL_PATH, N_PARAGRAPHS, TEMPERATURE,
)

setup_logging()
logger = logging.getLogger(__name__)

BASE_DIR = pathlib.Path(__file__).parent
CONTEXT_DIR = BASE_DIR / "Context"

THINKING_MODELS = {
    "DeepSeek-R1-Qwen3-8B",
    "DeepSeek-R1-Llama-8B",
    "DeepSeek-R1-Qwen-32B",
    "DeepSeek-R1-Llama-70B",
}

THINKING_DISABLED_MODELS = {
    "Qwen3.5-9B",
}

# Models that produce reasoning/chain-of-thought output.
# GPT-OSS-20B uses its own thinking format handled by parse_thinking_output.
REASONING_MODELS = THINKING_MODELS | {"GPT-OSS-20B"}

INVALID_PROMPTS = {"C1", "C2", "C3", "...", "{generatedadversarialprompt}", "generated prompt"}


def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

_UNICODE_TO_BYTE = {v: k for k, v in _bytes_to_unicode().items()}

def _fix_bpe_chars(text):
    try:
        byte_seq = bytearray()
        for char in text:
            if char in _UNICODE_TO_BYTE:
                byte_seq.append(_UNICODE_TO_BYTE[char])
            else:
                byte_seq.extend(char.encode('utf-8'))
        return byte_seq.decode('utf-8', errors='replace')
    except Exception:
        return text


class ThinkingClient:
    def __init__(self, model_path, enable_thinking=True):
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, clean_up_tokenization_spaces=False
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.generation_config.max_length = None

    def __call__(self, messages, max_new_tokens=32768, repetition_penalty=REPETITION_PENALTY):
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        target_device = next(self.model.parameters()).device
        inputs = self.tokenizer(text, return_tensors="pt").to(target_device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=TEMPERATURE,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][input_length:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return [{"generated_text": messages + [{"role": "assistant", "content": _fix_bpe_chars(result)}]}]


def _make_pipeline(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, clean_up_tokenization_spaces=False
    )
    pipe = hf_pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    pipe.model.generation_config.max_length = None
    return pipe


def _extract_json_objects(text):
    """
    Extract JSON objects from text using brace-depth tracking.
    More robust than regex: correctly handles nested structures and
    list values inside objects (e.g. "criteria_targeted": ["C1", "C2"]).
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


def load_system_prompt(scenario, criterion):
    prompt_file = BASE_DIR / scenario["dir"] / f"{scenario['prompt_prefix']}_{criterion}.txt"
    if not prompt_file.exists():
        logger.warning("Prompt file %s does not exist.", prompt_file)
        return None
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    # {n_prompts} replacement removed — target count is now defined
    # directly in the system prompt files.
    return content


def format_prompt(prompt):
    if isinstance(prompt, dict):
        return prompt.get("adversarial_prompt", prompt.get("prompt", str(prompt)))
    return str(prompt)


def generate_batch(scenario, criterion, cf_content, summary_content, paragraph_content,
                   cf_filename, client, generated_so_far=None):
    system_prompt = load_system_prompt(scenario, criterion)
    if system_prompt is None:
        return []

    user_message = build_user_message(
        scenario, cf_content, summary_content, paragraph_content, cf_filename
    )
    final_user_message = (
        f"Use the following {scenario['context_type']} as the context to generate adversarial prompts:\n"
        + user_message
    )
    if generated_so_far:
        prompt_list = "\n".join(f'"{p}"' for p in generated_so_far)
        final_user_message += f"\n\nDo NOT generate any of these already-generated prompts:\n{prompt_list}"

    max_tokens = 32768 if isinstance(client, ThinkingClient) else 2048
    try:
        response = client(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": final_user_message}],
            repetition_penalty=REPETITION_PENALTY,
            max_new_tokens=max_tokens,
        )
    except Exception as e:
        if "roles must alternate" in str(e):
            logger.warning(
                "Model %s does not support system role, merging into user message.",
                type(client).__name__,
            )
            response = client(
                [{"role": "user", "content": f"{system_prompt}\n\n{final_user_message}"}],
                repetition_penalty=REPETITION_PENALTY,
                max_new_tokens=max_tokens,
            )
        else:
            raise

    raw_content = response[0]["generated_text"][-1]["content"].strip()
    parsed_output = parse_thinking_output(raw_content)
    result_text = parsed_output["answer"]
    # batch_reasoning is the chain-of-thought block (None for non-reasoning models)
    batch_reasoning = parsed_output["thinking"]

    logger.info("Raw output before parsing:\n%s", result_text)

    results = []

    # Parser 1: JSON array
    start, end = result_text.find('['), result_text.rfind(']')
    if start != -1 and end != -1 and '{' in result_text[start:end]:
        try:
            results = json.loads(result_text[start:end + 1])
            if results:
                logger.info("Parser: JSON array (%d items)", len(results))
        except json.JSONDecodeError:
            pass

    # Parser 2: individual JSON objects via brace-depth tracking
    # Handles nested structures and list values (e.g. "criteria_targeted": ["C1"])
    if not results:
        results = _extract_json_objects(result_text)
        if results:
            logger.info("Parser: JSON objects (%d items)", len(results))

    # Parser 3: prose fallback
    if not results:
        for match in re.finditer(r'"([^"\n]{20,500})"', result_text):
            matched_text = match.group(1).strip()
            if '**' not in matched_text and not matched_text.startswith((']', '-', '\n')):
                results.append({"prompt": matched_text})
        if results:
            logger.info("Parser: prose fallback (%d items)", len(results))

    # Parser 4: markdown list fallback
    if not results:
        for match in re.finditer(r'\*\*Prompt \d+.*?\*\*\s*\n\s*"([^"]{20,})"',
                                  result_text, re.DOTALL):
            results.append({"prompt": match.group(1).strip()})
        if results:
            logger.info("Parser: markdown list fallback (%d items)", len(results))

    if not results:
        logger.warning("All parsers failed. Raw output (first 500 chars): %s", result_text[:500])

    normalized = []
    for r in results:
        if isinstance(r, str):
            r = {"prompt": r}
        if isinstance(r, dict):
            # Attach the batch reasoning to each prompt so it can be saved per row
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
            "Filtered %d/%d items (invalid or outside word count bounds %d-%d): %s",
            len(normalized) - len(filtered), len(normalized),
            MIN_WORD_COUNT, MAX_WORD_COUNT,
            [format_prompt(r) for r in normalized if r not in filtered],
        )

    return filtered


def accumulate_prompts(scenario, criterion, cf_content, summary_content, paragraph_content,
                       cf_filename, client, generated_so_far=None):
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
            scenario, criterion, cf_content, summary_content, paragraph_content,
            cf_filename, client, generated_so_far=generated_so_far,
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


def preprocess_form(cf_dir, client, cf_stem):
    cf_file = cf_dir / f"{cf_dir.name}.txt"
    with open(cf_file, 'r', encoding='utf-8') as f:
        cf_content = f.read()
    cf_content = fix_bold_headings(cf_content)
    if not cf_content.startswith("Consent Form:"):
        cf_content = f"Consent Form:\n\n{cf_content}"
        save_file(cf_content, cf_file)
    logger.info("Generating summary...")
    summary_content = generate_summary(cf_content, client)
    if summary_content is None:
        raise RuntimeError(f"Summary content could not be generated for {cf_dir.name}")
    summary_with_label = f"Consent Form Summary:\n\n{summary_content}"
    save_file(summary_with_label, cf_dir / f"{cf_stem}.SUM.txt")
    logger.info("Generating paragraphs...")
    paragraphs = [
        f"Extracted Paragraph from Consent Form:\n\n{par}"
        for par in generate_paragraph(cf_content)[:N_PARAGRAPHS]
    ]
    for i, par in enumerate(paragraphs, start=1):
        save_file(par, cf_dir / f"{cf_stem}.PAR{i}.txt")
        save_file(f"{summary_with_label}\n\n{par}", cf_dir / f"{cf_stem}.SUM_PAR{i}.txt")
    return cf_content, summary_with_label, paragraphs


def run_for_model(cf_content, summary_content, paragraphs, cf_filename, client):
    model_results = {}
    # Shared across all scenarios to prevent duplicate prompts in the output.
    generated_so_far = set()
    for scenario in SCENARIOS:
        logger.info("Running scenario: %s", scenario["scenario_id"])
        if scenario["needs_paragraph"]:
            for i, par in enumerate(paragraphs, start=1):
                label = f"{scenario['scenario_id']} (PAR{i})"
                model_results[label] = {}
                for criterion in CRITERIA:
                    prompts = accumulate_prompts(
                        scenario, criterion, cf_content, summary_content, par,
                        cf_filename, client, generated_so_far=generated_so_far,
                    )
                    model_results[label][criterion] = prompts
                    generated_so_far.update(format_prompt(p) for p in prompts)
        elif scenario["needs_sum_par"]:
            for i, par in enumerate(paragraphs, start=1):
                label = f"{scenario['scenario_id']} (PAR{i})"
                model_results[label] = {}
                for criterion in CRITERIA:
                    prompts = accumulate_prompts(
                        scenario, criterion, cf_content,
                        f"{summary_content}\n\n{par}", None,
                        cf_filename, client, generated_so_far=generated_so_far,
                    )
                    model_results[label][criterion] = prompts
                    generated_so_far.update(format_prompt(p) for p in prompts)
        else:
            label = scenario["scenario_id"]
            model_results[label] = {}
            for criterion in CRITERIA:
                prompts = accumulate_prompts(
                    scenario, criterion, cf_content, summary_content, None,
                    cf_filename, client, generated_so_far=generated_so_far,
                )
                model_results[label][criterion] = prompts
                generated_so_far.update(format_prompt(p) for p in prompts)
    return model_results


def save_model_csv(model_results, model_name, cf_stem, is_reasoning_model):
    rows = []
    for label, criteria_dict in model_results.items():
        for criterion, prompts in criteria_dict.items():
            for i, prompt_item in enumerate(prompts, start=1):
                prompt_text = format_prompt(prompt_item)
                reasoning = None
                if is_reasoning_model and isinstance(prompt_item, dict):
                    reasoning = prompt_item.get("_reasoning")
                rows.append({
                    "consent_form": cf_stem,
                    "scenario": label,
                    "criterion": criterion,
                    "prompt_index": i,
                    "model_name": model_name,
                    "data_source": prompt_item.get("data_source") if isinstance(prompt_item, dict) else None,
                    "prompt": prompt_text,
                    "reasoning": reasoning,
                })

    if not rows:
        logger.warning("No results to save for model %s on %s.", model_name, cf_stem)
        return

    # Sanitize model name for use in filename
    safe_model_name = re.sub(r'[^\w\-]', '_', model_name)
    csv_path = BASE_DIR / f"{safe_model_name}_results.csv"
    new_df = pd.DataFrame(rows)

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        key_cols = ["consent_form", "scenario", "criterion", "prompt_index", "model_name"]
        combined.drop_duplicates(subset=key_cols, keep='last', inplace=True)
        combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
    else:
        new_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    logger.info("Saved %d rows for %s to %s", len(rows), model_name, csv_path)


def validate_word_count(sentence: str) -> str | None:
    if not isinstance(sentence, str):
        if sentence is not None:
            logger.error("Invalid Data Type: expected str, got %s", type(sentence).__name__)
        return None
    word_count = len(sentence.split())
    if MIN_WORD_COUNT <= word_count <= MAX_WORD_COUNT:
        return sentence
    return None


if __name__ == "__main__":
    if CONTEXT_DIR.exists():
        for consent_form_name in CONTEXT_DIR.iterdir():
            if not consent_form_name.is_dir():
                continue
            cf_dir = consent_form_name
            cf_stem = cf_dir.name

            sum_file = cf_dir / f"{cf_stem}.SUM.txt"
            par1_file = cf_dir / f"{cf_stem}.PAR1.txt"
            cf_file = cf_dir / f"{cf_stem}.txt"

            if cf_file.exists() and sum_file.exists() and par1_file.exists():
                logger.info("=== Preprocessing files found, loading from disk ===")
                with open(cf_file, 'r', encoding='utf-8') as f:
                    cf_content = f.read()
                with open(sum_file, 'r', encoding='utf-8') as f:
                    summary_content = f.read()
                paragraphs = []
                for i in range(1, N_PARAGRAPHS + 1):
                    par_file = cf_dir / f"{cf_stem}.PAR{i}.txt"
                    if par_file.exists():
                        with open(par_file, 'r', encoding='utf-8') as f:
                            paragraphs.append(f.read())
            else:
                logger.info("=== Preprocessing with sum model ===")
                sum_client = _make_pipeline(SUM_MODEL_PATH)
                try:
                    cf_content, summary_content, paragraphs = preprocess_form(
                        cf_dir, sum_client, cf_stem
                    )
                except RuntimeError as e:
                    logger.error("Skipping %s: %s", cf_stem, e)
                    del sum_client
                    torch.cuda.empty_cache()
                    continue
                del sum_client
                torch.cuda.empty_cache()

            for model_name, model_path in AP_MODEL_PATHS.items():
                client = None
                try:
                    logger.info("=== Running %s ===", model_name)
                    if model_name in THINKING_MODELS:
                        client = ThinkingClient(model_path, enable_thinking=True)
                    elif model_name in THINKING_DISABLED_MODELS:
                        client = ThinkingClient(model_path, enable_thinking=False)
                    else:
                        client = _make_pipeline(model_path)

                    model_results = run_for_model(
                        cf_content, summary_content, paragraphs, f"{cf_stem}.txt", client
                    )
                    del client
                    torch.cuda.empty_cache()

                    is_reasoning = model_name in REASONING_MODELS
                    save_model_csv(model_results, model_name, cf_stem, is_reasoning)

                except Exception as e:
                    logger.error("=== FAILED %s: %s ===", model_name, e, exc_info=True)
                    if client is not None:
                        del client
                    torch.cuda.empty_cache()

        logger.info("Done!")
    else:
        logger.error("CONTEXT_DIR does not exist: %s", CONTEXT_DIR)