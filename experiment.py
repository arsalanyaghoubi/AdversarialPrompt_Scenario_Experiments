import gc
import json
import logging
import pathlib
import re
import sys
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline as hf_pipeline
from pipeline_code.preprocess import run_preprocess
from pipeline_code.prompt_gen import CRITERIA, SCENARIOS, accumulate_prompts, format_prompt
from utils.log_config import setup_logging
from utils.config import ( REPETITION_PENALTY,AP_MODEL_PATHS, SUM_MODEL_PATH, TEMPERATURE, TASK, SKIP_SUMMARY_GEN)

setup_logging()
logger = logging.getLogger(__name__)

BASE_DIR     = pathlib.Path(__file__).parent
CONTEXT_DIR  = BASE_DIR / "Context"

# ── Task validation ────────────────────────────────────────────────────────────
TASK_RUNS_PREPROCESS = {"preprocess", "full_pipeline"}
TASK_RUNS_AUTOPROMPT = {"prompt_generation", "full_pipeline"}
VALID_TASKS          = TASK_RUNS_PREPROCESS | TASK_RUNS_AUTOPROMPT

# Fail immediately at startup before any GPU or file work begins.
if TASK not in VALID_TASKS:
    raise ValueError(
        f"Unknown task: '{TASK}' in config.json. "
        f"Must be one of: {sorted(VALID_TASKS)}"
    )

# ── Model classification sets ──────────────────────────────────────────────────
# ThinkingClient with enable_thinking=True
THINKING_MODELS = {
    "DeepSeek-R1-Qwen3-8B",
    "DeepSeek-R1-Llama-8B",
    "DeepSeek-R1-Qwen-32B",
    "DeepSeek-R1-Llama-70B",
}
# ThinkingClient with enable_thinking=False
THINKING_DISABLED_MODELS = {
    "Qwen3.5-9B",
}
# ThinkingClient with skip_special_tokens=False to preserve
# <|channel|> markers for parse_thinking_output.
GPT_OSS_MODELS = {
    "GPT-OSS-20B",
}
# All models that produce reasoning/chain-of-thought output
REASONING_MODELS = THINKING_MODELS | GPT_OSS_MODELS

# Seconds to wait after fully unloading a model before loading the next.
MODEL_UNLOAD_WAIT = 10
MODEL_LOAD_WAIT = 5


# ── BPE character fixing ───────────────────────────────────────────────────────

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


# ── Model clients ──────────────────────────────────────────────────────────────

class ThinkingClient:
    """
    Direct HuggingFace generation client for models that expose reasoning blocks
    (DeepSeek R1, GPT-OSS).

    MAX_NEW_TOKENS is read by generate_batch in prompt_gen.py via
    getattr(client, 'MAX_NEW_TOKENS', 2048) to avoid a circular import.
    """
    MAX_NEW_TOKENS = 32768

    def __init__(self, model_path, enable_thinking=True, skip_special_tokens=True):
        self.enable_thinking    = enable_thinking
        # skip_special_tokens=False preserves <|channel|> markers for GPT-OSS.
        self.skip_special_tokens = skip_special_tokens
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
        # Clears the max_length=20 HuggingFace default so max_new_tokens is
        # the only generation limit.
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
            # Model does not support enable_thinking parameter.
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # next(parameters()).device is reliable with device_map="auto" sharding;
        # self.model.device can return "meta" in that case.
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
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=self.skip_special_tokens)
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


def _unload(client):
    """
    Release a model client and free GPU memory.
    IMPORTANT: always also delete the caller's variable after this call.
    _unload only removes its own local copy of the reference; the caller's
    variable must be set to None or deleted too or the model stays in VRAM.
    """
    if client is not None:
        del client
    gc.collect()
    torch.cuda.empty_cache()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_cf_data(cf_dirs):
    """
    Load all preprocessed consent-form files from disk into memory.
    Warns and skips any CF directory that is missing required files.
    Isolates SUM_PAR files before PAR files to avoid overlap in filename matching.
    """
    consent_form_data = {}

    for cf_dir in cf_dirs:
        cf_stem    = cf_dir.name
        file_paths = [f for f in cf_dir.iterdir() if f.is_file()]

        # Isolate SUM_PAR files first to prevent them matching .PAR filter below.
        sum_par_files = sorted(
            [f for f in file_paths if ".SUM_PAR" in f.name],
            key=lambda p: int(m.group(1)) if (m := re.search(r'\.SUM_PAR(\d+)', p.name)) else 0,
        )
        sum_files = [
            f for f in file_paths
            if (".SUM." in f.name or f.name.endswith(".SUM.txt")) and f not in sum_par_files
        ]
        par_files = sorted(
            [f for f in file_paths if ".PAR" in f.name and f not in sum_par_files],
            key=lambda p: int(m.group(1)) if (m := re.search(r'\.PAR(\d+)', p.name)) else 0,
        )
        main_files = [
            f for f in file_paths
            if f.suffix == ".txt"
            and f not in sum_files
            and f not in par_files
            and f not in sum_par_files
        ]

        if not (main_files and sum_files and par_files):
            logger.warning(
                "[%s] Missing required preprocessed files — skipping. "
                "Run with task=preprocess or task=full_pipeline first.",
                cf_stem,
            )
            continue

        consent_form_data[cf_stem] = {
            "cf_content":       main_files[0].read_text(encoding='utf-8'),
            "summary_content":  sum_files[0].read_text(encoding='utf-8'),
            "paragraphs":       [p.read_text(encoding='utf-8') for p in par_files],
            "summary_paragraphs": [p.read_text(encoding='utf-8') for p in sum_par_files],
        }
        logger.info(
            "[%s] Loaded from disk — %d PAR, %d SUM_PAR files.",
            cf_stem, len(par_files), len(sum_par_files),
        )

    return consent_form_data


# ── Inference orchestration ────────────────────────────────────────────────────

def _run_for_model(cf_content, summary_content, paragraphs, summary_paragraphs,
                   cf_filename, client):
    """
    Run all scenarios for one (model, consent-form) combination.
    Per-scenario errors are caught and logged so partial results are preserved.

    generated_so_far is shared across all scenarios and criteria to prevent
    duplicate prompts appearing anywhere in the output for this CF.
    """
    model_results  = {}
    generated_so_far = set()

    for scenario in SCENARIOS:
        logger.info("Running scenario: %s", scenario["scenario_id"])
        try:
            if scenario["needs_paragraph"]:
                # Scenario 4 — PAR: iterate over individual paragraphs
                for i, par in enumerate(paragraphs, start=1):
                    label = f"{scenario['scenario_id']} (PAR{i})"
                    model_results[label] = {}
                    for criterion in CRITERIA:
                        prompts = accumulate_prompts(
                            scenario, criterion, par, cf_filename, client,
                            generated_so_far=generated_so_far,
                        )
                        model_results[label][criterion] = prompts
                        generated_so_far.update(format_prompt(p) for p in prompts)

            elif scenario["needs_sum_par"]:
                # Scenario 3 — SUM_PAR: use pre-built combined files
                for i, sum_par in enumerate(summary_paragraphs, start=1):
                    label = f"{scenario['scenario_id']} (SUM_PAR{i})"
                    model_results[label] = {}
                    for criterion in CRITERIA:
                        prompts = accumulate_prompts(
                            scenario, criterion, sum_par, cf_filename, client,
                            generated_so_far=generated_so_far,
                        )
                        model_results[label][criterion] = prompts
                        generated_so_far.update(format_prompt(p) for p in prompts)

            elif scenario["needs_summary"]:
                # Scenario 2 — CF_SUM: whole summary as context
                label = scenario["scenario_id"]
                model_results[label] = {}
                for criterion in CRITERIA:
                    prompts = accumulate_prompts(
                        scenario, criterion, summary_content, cf_filename, client,
                        generated_so_far=generated_so_far,
                    )
                    model_results[label][criterion] = prompts
                    generated_so_far.update(format_prompt(p) for p in prompts)

            else:
                # Scenario 1 — ConsentForm (needs_cf=True): full CF text as context
                label = scenario["scenario_id"]
                model_results[label] = {}
                for criterion in CRITERIA:
                    prompts = accumulate_prompts(
                        scenario, criterion, cf_content, cf_filename, client,
                        generated_so_far=generated_so_far,
                    )
                    model_results[label][criterion] = prompts
                    generated_so_far.update(format_prompt(p) for p in prompts)

        except Exception as e:
            logger.error(
                "Scenario %s failed: %s — skipping, preserving other results.",
                scenario["scenario_id"], e, exc_info=True,
            )
            continue

    return model_results


def run_autoprompt(consent_form_data):
    """
    Phase 2: Load each AP model once, run it over ALL consent forms, then unload.
    Load/unload cycles = number of AP models (not CFs × models).
    """
    model_names = list(AP_MODEL_PATHS.keys())
    for model_idx, (model_name, model_path) in enumerate(AP_MODEL_PATHS.items()):
        client = None
        logger.info(
            "=== Loading model %d/%d: %s ===",
            model_idx + 1, len(model_names), model_name,
        )
        try:
            if model_name in THINKING_MODELS:
                client = ThinkingClient(model_path, enable_thinking=True)
            elif model_name in THINKING_DISABLED_MODELS:
                client = ThinkingClient(model_path, enable_thinking=False)
            elif model_name in GPT_OSS_MODELS:
                # skip_special_tokens=False preserves <|channel|> markers so
                # parse_thinking_output can extract the reasoning block.
                client = ThinkingClient(
                    model_path, enable_thinking=False, skip_special_tokens=False
                )
            else:
                client = _make_pipeline(model_path)

            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Model loaded. Waiting 5s before starting inference...")
            time.sleep(MODEL_LOAD_WAIT)

            is_reasoning = model_name in REASONING_MODELS

            for cf_stem, data in consent_form_data.items():
                logger.info("[%s] Running %s...", cf_stem, model_name)
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    model_results = _run_for_model(
                        data["cf_content"],
                        data["summary_content"],
                        data["paragraphs"],
                        data["summary_paragraphs"],
                        f"{cf_stem}.txt",
                        client,
                    )
                    save_model_csv(model_results, model_name, cf_stem, is_reasoning)
                except Exception as e:
                    logger.error(
                        "[%s] %s failed: %s", cf_stem, model_name, e, exc_info=True
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue

        except Exception as e:
            logger.error("Failed to load %s: %s", model_name, e, exc_info=True)

        finally:
            logger.info("=== Unloading %s ===", model_name)
            _unload(client)
            client = None  # release caller's reference so VRAM is fully freed

        if model_idx < len(model_names) - 1:
            logger.info("Waiting %ds before loading next model...", MODEL_UNLOAD_WAIT)
            time.sleep(MODEL_UNLOAD_WAIT)


# ── Output ─────────────────────────────────────────────────────────────────────

def save_model_csv(model_results, model_name, cf_stem, is_reasoning_model):
    """
    Append results for one (model, CF) pair to a per-model CSV file.
    Deduplicates on (consent_form, scenario, criterion, prompt_index, model_name)
    so re-runs don't create duplicates.
    """
    rows = []
    for label, criteria_dict in model_results.items():
        for criterion, prompts in criteria_dict.items():
            for i, prompt_item in enumerate(prompts, start=1):
                prompt_text = format_prompt(prompt_item)
                reasoning   = None
                data_source = None
                if isinstance(prompt_item, dict):
                    data_source = prompt_item.get("data_source")
                    if is_reasoning_model:
                        reasoning = prompt_item.get("_reasoning")
                rows.append({
                    "consent_form": cf_stem,
                    "scenario":     label,
                    "criterion":    criterion,
                    "prompt_index": i,
                    "model_name":   model_name,
                    "prompt":       prompt_text,
                    "data_source":  data_source,
                    "reasoning":    reasoning,
                })

    if not rows:
        logger.warning("No results to save for model %s on %s.", model_name, cf_stem)
        return

    safe_model_name = re.sub(r'[^\w\-]', '_', model_name)
    csv_path = BASE_DIR / f"{safe_model_name}_results.csv"
    new_df   = pd.DataFrame(rows)

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
        combined    = pd.concat([existing_df, new_df], ignore_index=True)
        key_cols    = ["consent_form", "scenario", "criterion", "prompt_index", "model_name"]
        combined.drop_duplicates(subset=key_cols, keep='last', inplace=True)
        combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
    else:
        new_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    logger.info("Saved %d rows for %s / %s to %s", len(rows), model_name, cf_stem, csv_path)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not CONTEXT_DIR.exists():
        logger.error("CONTEXT_DIR does not exist: %s", CONTEXT_DIR)
        sys.exit(1)

    cf_dirs = sorted([d for d in CONTEXT_DIR.iterdir() if d.is_dir()])
    if not cf_dirs:
        logger.error("No consent form directories found in %s", CONTEXT_DIR)
        sys.exit(1)
    logger.info("Found %d consent form(s): %s", len(cf_dirs), [d.name for d in cf_dirs])

    # ── Phase 1: Preprocessing ─────────────────────────────────────────────────
    # Sum model is loaded only when the task explicitly requires preprocessing.
    if TASK in TASK_RUNS_PREPROCESS:
        logger.info("=== Phase 1: Preprocessing (task=%s, skip_summary_gen=%s) ===",
                    TASK, SKIP_SUMMARY_GEN)
        sum_client = _make_pipeline(SUM_MODEL_PATH)
        run_preprocess(sum_client, skip_if_exists=SKIP_SUMMARY_GEN)
        logger.info("=== Preprocessing complete. Unloading sum model. ===")
        _unload(sum_client)
        del sum_client   # release caller's reference so VRAM is freed before AP models load
    else:
        logger.info("=== Skipping preprocessing (task=%s) ===", TASK)

    # ── Phase 2: Prompt generation ─────────────────────────────────────────────
    # Each AP model loads once, processes every CF, then unloads.
    # Sum model is never touched in this phase.
    if TASK in TASK_RUNS_AUTOPROMPT:
        logger.info("=== Phase 2: Prompt generation (task=%s) ===", TASK)
        consent_form_data = load_cf_data(cf_dirs)
        if not consent_form_data:
            logger.error(
                "No preprocessed CFs available. "
                "Run with task=preprocess or task=full_pipeline first."
            )
            sys.exit(1)
        logger.info(
            "%d CF(s) ready for inference: %s",
            len(consent_form_data), list(consent_form_data.keys()),
        )
        run_autoprompt(consent_form_data)
    else:
        logger.info("=== Skipping prompt generation (task=%s) ===", TASK)

    logger.info("Done!")