import sys, os, json, pathlib, logging, re
import torch
import pandas as pd
from transformers import pipeline as hf_pipeline
from adversarial_pipeline.pipeline.preprocess import fix_bold_headings, generate_summary, generate_paragraph, save_file
from adversarial_pipeline.pipeline.prompt_gen import build_user_message, SCENARIOS, CRITERIA
from adversarial_pipeline.utils.logging import setup_logging
from adversarial_pipeline.utils.thinking import parse_thinking_output

setup_logging()
logger = logging.getLogger(__name__)

BASE_DIR = pathlib.Path(__file__).parent
REPO_DIR = BASE_DIR.parent
CONTEXT_DIR = REPO_DIR / "Context"

with open(BASE_DIR / "config.json") as f:
    CONFIG = json.load(f)

CF_STEM = CONFIG["cf_stem"]
AP_MODEL_PATHS = CONFIG["ap_model_paths"]
SUM_MODEL_PATH = CONFIG["sum_model_path"]
N_PARAGRAPHS = CONFIG["n_paragraphs"]


THINKING_MODELS = {"Qwen3.5-9B", "DeepSeek-R1-Qwen3-8B", "DeepSeek-R1-Llama-8B", "DeepSeek-R1-Qwen-32B", "DeepSeek-R1-Llama-70B"}


class ThinkingClient:
    """HF transformers client with proper enable_thinking chat template support."""
    def __init__(self, model_path, enable_thinking=True):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    def __call__(self, messages, max_new_tokens=8192):
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        if hasattr(self.tokenizer, 'byte_decoder'):
            try:
                byte_seq = bytearray()
                for char in result:
                    if char in self.tokenizer.byte_decoder:
                        byte_seq.append(self.tokenizer.byte_decoder[char])
                    else:
                        byte_seq.extend(char.encode('utf-8'))
                result = byte_seq.decode('utf-8', errors='replace')
            except Exception:
                result = result.replace('\u0120', ' ').replace('\u010a', '\n')
        return [{"generated_text": messages + [{"role": "assistant", "content": result}]}]


def load_system_prompt(scenario, criterion):
    prompt_file = REPO_DIR / scenario["dir"] / f"{scenario['prompt_prefix']}_{criterion}.txt"
    if not prompt_file.exists():
        logger.warning("Prompt file %s does not exist.", prompt_file)
        return None
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    return content.replace("{n_prompts}", str(scenario["target"]))


def generate_batch(scenario, criterion, cf_content, summary_content, paragraph_content, cf_filename, client):
    system_prompt = load_system_prompt(scenario, criterion)
    if system_prompt is None:
        return []
    user_message = build_user_message(scenario, cf_content, summary_content, paragraph_content, cf_filename)
    user_request = f"Use the following {scenario['context_type']} as the context to generate adversarial prompts:\n"
    final_user_message = user_request + user_message
    try:
        response = client(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_user_message}
            ],
            max_new_tokens=8192
        )
    except Exception as e:
        if "roles must alternate" in str(e):
            # Mistral and some models don't support system role — merge into user message
            response = client(
                [{"role": "user", "content": f"{system_prompt}\n\n{final_user_message}"}],
                max_new_tokens=8192
            )
        else:
            raise
    result_text = response[0]["generated_text"][-1]["content"].strip()
    result_text = parse_thinking_output(result_text)["answer"]
    results = []
    start = result_text.find('[')
    end = result_text.rfind(']')
    if start != -1 and end != -1:
        try:
            results = json.loads(result_text[start:end+1])
        except json.JSONDecodeError:
            pass
    if not results:
        for match in re.finditer(r'\{.*?\}', result_text, re.DOTALL):
            try:
                results.append(json.loads(match.group()))
            except json.JSONDecodeError:
                continue
    INVALID_PROMPTS = {"C1", "C2", "C3", "...", "{generatedadversarialprompt}", "generated prompt"}
    results = [
        r for r in results
        if isinstance(r, dict) and len(format_prompt(r)) > 10 and format_prompt(r) not in INVALID_PROMPTS
    ]
    return results


def accumulate_prompts(scenario, criterion, cf_content, summary_content, paragraph_content, cf_filename, client):
    accumulated = []
    max_retries = 3
    retries = 0
    while len(accumulated) < scenario["target"]:
        if retries >= max_retries:
            logger.warning("Max retries reached for %s %s, skipping.", scenario["scenario_id"], criterion)
            break
        new_prompts = generate_batch(scenario, criterion, cf_content, summary_content, paragraph_content, cf_filename, client)
        if not new_prompts:
            retries += 1
        else:
            retries = 0
            accumulated.extend(new_prompts)
    return accumulated[:scenario["target"]]


def preprocess_form(cf_dir, client):
    cf_file = cf_dir / f"{cf_dir.name}.txt"
    with open(cf_file, 'r', encoding='utf-8') as f:
        cf_content = f.read()
    cf_content = fix_bold_headings(cf_content)
    if not cf_content.startswith("Consent Form:"):
        cf_content = f"Consent Form:\n\n{cf_content}"
        save_file(cf_content, cf_file)
    logger.info("Generating summary...")
    summary_content = generate_summary(cf_content, client)
    summary_with_label = f"Consent Form Summary:\n\n{summary_content}"
    save_file(summary_with_label, cf_dir / f"{CF_STEM}.SUM.txt")
    logger.info("Generating paragraphs...")
    paragraph_content = generate_paragraph(cf_content)
    paragraphs = [f"Extracted Paragraph from Consent Form:\n\n{par}" for par in paragraph_content[:N_PARAGRAPHS]]
    for i, par in enumerate(paragraphs, start=1):
        save_file(par, cf_dir / f"{CF_STEM}.PAR{i}.txt")
        save_file(f"{summary_with_label}\n\n{par}", cf_dir / f"{CF_STEM}.SUM_PAR{i}.txt")
    return cf_content, summary_with_label, paragraphs


def run_for_model(cf_content, summary_content, paragraphs, cf_filename, client):
    model_results = {}
    for scenario in SCENARIOS:
        logger.info("Running scenario: %s", scenario["scenario_id"])
        if scenario["needs_paragraph"]:
            for i, par in enumerate(paragraphs, start=1):
                label = f"{scenario['scenario_id']} (PAR{i})"
                model_results[label] = {}
                for criterion in CRITERIA:
                    results = accumulate_prompts(scenario, criterion, cf_content, summary_content, par, cf_filename, client)
                    model_results[label][criterion] = results
        elif scenario.get("needs_sum_par"):
            for i, par in enumerate(paragraphs, start=1):
                label = f"{scenario['scenario_id']} (PAR{i})"
                model_results[label] = {}
                sum_par_content = f"{summary_content}\n\n{par}"
                for criterion in CRITERIA:
                    results = accumulate_prompts(scenario, criterion, cf_content, sum_par_content, None, cf_filename, client)
                    model_results[label][criterion] = results
        else:
            label = scenario["scenario_id"]
            model_results[label] = {}
            for criterion in CRITERIA:
                results = accumulate_prompts(scenario, criterion, cf_content, summary_content, None, cf_filename, client)
                model_results[label][criterion] = results
    return model_results


def format_prompt(prompt):
    if isinstance(prompt, dict):
        return prompt.get("adversarial_prompt", prompt.get("prompt", str(prompt)))
    return str(prompt)


def save_comparison_csv(all_results):
    model_names = [m for m in all_results if all_results[m]]
    if not model_names:
        logger.warning("No completed model results to save.")
        return
    scenario_labels = list(all_results[model_names[0]].keys())
    rows = []
    for label in scenario_labels:
        for criterion in CRITERIA:
            max_prompts = max(
                len(all_results[m].get(label, {}).get(criterion, []))
                for m in model_names
            )
            for i in range(max_prompts):
                row = {
                    "consent_form": CF_STEM,
                    "scenario": label,
                    "criterion": criterion,
                    "prompt_index": i + 1,
                }
                for model_name in model_names:
                    prompts = all_results[model_name].get(label, {}).get(criterion, [])
                    row[model_name] = format_prompt(prompts[i]) if i < len(prompts) else ""
                rows.append(row)

    csv_path = BASE_DIR / f"{CF_STEM}_comparison_updated.csv"
    new_df = pd.DataFrame(rows)
    key_cols = ["consent_form", "scenario", "criterion", "prompt_index"]

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
        existing_df = existing_df.set_index(key_cols)
        new_df = new_df.set_index(key_cols)
        for col in new_df.columns:
            existing_df[col] = new_df[col]
        existing_df.reset_index().to_csv(csv_path, index=False, encoding='utf-8-sig')
    else:
        new_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    logger.info("Saved comparison CSV to %s", csv_path)


if __name__ == "__main__":
    cf_dir = CONTEXT_DIR / CF_STEM
    if not cf_dir.exists():
        logger.error("Consent form directory not found: %s", cf_dir)
        sys.exit(1)

    sum_file = cf_dir / f"{CF_STEM}.SUM.txt"
    par1_file = cf_dir / f"{CF_STEM}.PAR1.txt"
    if sum_file.exists() and par1_file.exists():
        logger.info("=== Preprocessing files found, loading from disk ===")
        cf_file = cf_dir / f"{CF_STEM}.txt"
        with open(cf_file, 'r', encoding='utf-8') as f:
            cf_content = f.read()
        with open(sum_file, 'r', encoding='utf-8') as f:
            summary_content = f.read()
        paragraphs = []
        for i in range(1, N_PARAGRAPHS + 1):
            par_file = cf_dir / f"{CF_STEM}.PAR{i}.txt"
            if par_file.exists():
                with open(par_file, 'r', encoding='utf-8') as f:
                    paragraphs.append(f.read())
    else:
        logger.info("=== Preprocessing with sum model ===")
        sum_client = hf_pipeline("text-generation", model=SUM_MODEL_PATH, device_map="auto")
        cf_content, summary_content, paragraphs = preprocess_form(cf_dir, sum_client)
        del sum_client
        torch.cuda.empty_cache()

    all_results = {}
    for model_name, model_path in AP_MODEL_PATHS.items():
        try:
            logger.info("=== Running %s ===", model_name)
            if model_name in THINKING_MODELS:
                client = ThinkingClient(model_path, enable_thinking=False)
            else:
                client = hf_pipeline("text-generation", model=model_path, device_map="auto")
            all_results[model_name] = run_for_model(cf_content, summary_content, paragraphs, f"{CF_STEM}.txt", client)
            del client
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error("=== FAILED %s: %s ===", model_name, e, exc_info=True)
            all_results[model_name] = {}
            torch.cuda.empty_cache()
        save_comparison_csv(all_results)

    logger.info("Done!")