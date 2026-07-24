import pathlib
import json
import os
import torch
from transformers import pipeline as hf_pipeline
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from preprocess import fix_bold_headings, generate_summary, generate_paragraph
from autoPrompt_generation import generate_batch, BASE_DIR, CONTEXT_DIR

with open(BASE_DIR / "config.json") as f:
    CONFIG = json.load(f)

N_PARAGRAPHS = CONFIG["n_paragraphs"]
CRITERIA = ["C1", "C2", "C3"]
OUTPUT_DIR = BASE_DIR
CF_STEM = "NCT04428203_001"

SCENARIOS = [
    {
        "dir": "1_OrigCF_Context",
        "scenario_id": "Original Consent Form",
        "needs_cf": True,
        "needs_summary": False,
        "needs_paragraph": False,
        "prompt_prefix": "CF_Adversarial_Prompt",
        "result_prefix": "OrigCF_Context",
        "target": CONFIG["targets"]["CF"],
    },
    {
        "dir": "2_SumCF_Context",
        "scenario_id": "Summary",
        "needs_cf": False,
        "needs_summary": True,
        "needs_paragraph": False,
        "prompt_prefix": "CF_SUM_Adversarial_Prompt",
        "result_prefix": "SumCF_Context",
        "target": CONFIG["targets"]["SUM"],
    },
    {
        "dir": "3_SUM_PARAG_Context",
        "scenario_id": "Summary + Paragraph",
        "needs_cf": False,
        "needs_summary": False,
        "needs_paragraph": False,
        "needs_sum_par": True,
        "prompt_prefix": "SUM_PAR_Adversarial_Prompt",
        "result_prefix": "SUM_PARAG_Context",
        "target": CONFIG["targets"]["SUM_PAR"],
    },
    {
        "dir": "4_PARAG_Context",
        "scenario_id": "Paragraph",
        "needs_cf": False,
        "needs_summary": False,
        "needs_paragraph": True,
        "prompt_prefix": "PARAG_Adversarial_Prompt",
        "result_prefix": "PARAG_Context",
        "target": CONFIG["targets"]["PAR"],
    },
]

MODEL_PATHS = {
    "Llama": "/home1/shared/Models/Llama/Llama-3.1-8B-Instruct",
    "Gemma": "/home1/shared/Models/Gemma/Gemma-3-12B-it",
    "Qwen": "/home1/shared/Models/Qwen/Qwen3.5-9B",
}

def preprocess_form(cf_dir, client):
    stem = cf_dir.name
    cf_file = cf_dir / f"{stem}.txt"
    with open(cf_file, 'r', encoding='utf-8') as f:
        cf_content = f.read()
    cf_content = fix_bold_headings(cf_content)
    if not cf_content.startswith("Consent Form:"):
        cf_content = f"Consent Form:\n\n{cf_content}"
    print("  Generating summary...")
    summary_content = generate_summary(cf_content, client)
    summary_with_label = f"Consent Form Summary:\n\n{summary_content}"
    print("  Generating paragraphs...")
    paragraph_content = generate_paragraph(cf_content)
    paragraphs = []
    for par in paragraph_content[:N_PARAGRAPHS]:
        paragraphs.append(f"Extracted Paragraph from Consent Form:\n\n{par}")
    print("  Preprocessing done.")
    return cf_content, summary_with_label, paragraphs

def accumulate_prompts(scenario, criterion, cf_content, summary_content, paragraph_content, cf_filename, client):
    accumulated = []
    max_retries = 20
    retries = 0
    while len(accumulated) < scenario["target"]:
        if retries >= max_retries:
            print(f"Max retries reached for {scenario['scenario_id']} {criterion}")
            break
        new_prompts = generate_batch(scenario, criterion, cf_content, summary_content, paragraph_content, cf_filename, client)
        if not new_prompts:
            retries += 1
        else:
            retries = 0
            accumulated.extend(new_prompts)
    return accumulated[:scenario["target"]]

def run_for_model(cf_content, summary_content, paragraphs, cf_filename, client):
    results = {}
    for criterion in CRITERIA:
        results[criterion] = {}

    for scenario in SCENARIOS:
        print(f"  Running scenario: {scenario['scenario_id']}")
        if scenario["needs_paragraph"]:
            for i, par in enumerate(paragraphs, start=1):
                label = f"{scenario['scenario_id']} (PAR{i})"
                for criterion in CRITERIA:
                    prompts = accumulate_prompts(scenario, criterion, cf_content, summary_content, par, cf_filename, client)
                    results[criterion][label] = prompts
        elif scenario.get("needs_sum_par"):
            for i, par in enumerate(paragraphs, start=1):
                sum_par_content = f"{summary_content}\n\n{par}"
                label = f"{scenario['scenario_id']} (PAR{i})"
                for criterion in CRITERIA:
                    prompts = accumulate_prompts(scenario, criterion, cf_content, sum_par_content, None, cf_filename, client)
                    results[criterion][label] = prompts
        else:
            label = scenario["scenario_id"]
            for criterion in CRITERIA:
                prompts = accumulate_prompts(scenario, criterion, cf_content, summary_content, None, cf_filename, client)
                results[criterion][label] = prompts

    return results

def format_prompt(prompt):
    if isinstance(prompt, dict):
        return prompt.get("adversarial_prompt", prompt.get("prompt", str(prompt)))
    return str(prompt)

def save_markdown(all_results, cf_name):
    first_model = list(MODEL_PATHS.keys())[0]
    scenario_labels = list(all_results[first_model][CRITERIA[0]].keys())

    for criterion in CRITERIA:
        lines = [f"# {criterion}", f"Consent Form: {cf_name}", ""]
        for scenario_label in scenario_labels:
            lines.append(f"## {scenario_label}")
            lines.append("")
            for model_name in MODEL_PATHS:
                prompts = all_results.get(model_name, {}).get(criterion, {}).get(scenario_label, [])
                lines.append(f"**{model_name}:**")
                if prompts:
                    for i, prompt in enumerate(prompts, start=1):
                        lines.append(f"    {i}. {format_prompt(prompt)}")
                else:
                    lines.append("    No prompts generated.")
                lines.append("")
        md_path = OUTPUT_DIR / f"{criterion}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"Saved {md_path}")

if __name__ == "__main__":
    device = int(os.environ.get("DEVICE", 5))
    cf_dir = CONTEXT_DIR / CF_STEM

    if not cf_dir.exists():
        print(f"Consent form directory not found: {cf_dir}")
        sys.exit(1)

    all_results = {}

    for model_name, model_path in MODEL_PATHS.items():
        print(f"\n=== Running {model_name} ===")
        client = hf_pipeline(
            "text-generation",
            model=model_path,
            device=device
        )
        cf_content, summary_content, paragraphs = preprocess_form(cf_dir, client)
        all_results[model_name] = run_for_model(cf_content, summary_content, paragraphs, f"{CF_STEM}.txt", client)
        del client
        torch.cuda.empty_cache()

    save_markdown(all_results, CF_STEM)
    print("\nDone! Markdown files saved to repo root.")