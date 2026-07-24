import logging
import pandas as pd
import os, json, pathlib, api_models
from transformers import pipeline as hf_pipeline
from context_generator import find_consent_forms, fix_bold_headings, generate_summary, generate_paragraph, save_file, CONTEXT_DIR
from prompt_generation import find_consent_form_pairs, validate_scenario, accumulate_prompts, save_results, SCENARIOS, CRITERIA
from adversarial_pipeline.utils.logging import setup_logging
from adversarial_pipeline.utils.io import encoder_

setup_logging()
logger = logging.getLogger(__name__)  # module-level, available everywhere


def run_preprocess(client):
    consent_forms = find_consent_forms(CONTEXT_DIR)[5:6]
    for cf_file in consent_forms:
        logger.info("Preprocessing %s...", cf_file.name)
        with open(cf_file, 'r', encoding='utf-8') as f:
            cf_content = f.read()
        cf_content = fix_bold_headings(cf_content)
        if not cf_content.startswith("Consent Form:"):
            cf_content = f"Consent Form:\n\n{cf_content}"
            save_file(cf_content, cf_file)
        try:
            summary_content = generate_summary(cf_content, client)
        except Exception as e:
            logger.exception("Error generating summary for %s: %s", cf_file.name, e)
            summary_content = None
        try:
            paragraph_content = generate_paragraph(cf_content)
        except Exception as e:
            logger.exception("Error generating paragraph for %s: %s", cf_file.name, e)
            paragraph_content = []
        if summary_content:
            save_file(f"Consent Form Summary:\n\n{summary_content}", cf_file.parent / f"{cf_file.stem}.SUM.txt")
        if paragraph_content:
            for i, par in enumerate(paragraph_content, start=1):
                par_filename = cf_file.parent / f"{cf_file.stem}.PAR{i}.txt"
                save_file(f"Extracted Paragraph from Consent Form:\n\n{par}", par_filename)
        else:
            logger.warning("Skipping empty paragraphs for %s", cf_file.name)
        if summary_content and paragraph_content:
            for i, par in enumerate(paragraph_content, start=1):
                combined = f"Consent Form Summary:\n\n{summary_content}\n\nExtracted Paragraph from Consent Form:\n\n{par}"
                sum_par_filename = cf_file.parent / f"{cf_file.stem}.SUM_PAR{i}.txt"
                save_file(combined, sum_par_filename)


def run_autoprompt(client):
    pairs = find_consent_form_pairs(CONTEXT_DIR)[5:6]
    for pair in pairs:
        if pair["cf"]:
            cf_content = pair["cf"].read_text(encoding="utf-8")
            cf_filename = pair["cf"].name
            cf_stem = pair["cf"].stem
        else:
            cf_content = None
            cf_filename = None
            cf_stem = None
        if pair["summary"]:
            summary_content = pair["summary"].read_text(encoding="utf-8")
        else:
            summary_content = None
        for scenario in SCENARIOS:
            if not validate_scenario(scenario, pair):
                continue
            if scenario["needs_paragraph"]:
                for i, par_file in enumerate(pair["paragraphs"], start=1):
                    paragraph_content = par_file.read_text(encoding="utf-8")
                    for criterion in CRITERIA:
                        results = accumulate_prompts(scenario, criterion, cf_content, summary_content, paragraph_content, cf_filename, client)
                        save_results(results, scenario, criterion, cf_stem, paragraph_index=i)
            elif scenario.get("needs_sum_par"):
                for i, sum_par_file in enumerate(pair["sum_paragraphs"], start=1):
                    sum_par_content = sum_par_file.read_text(encoding="utf-8")
                    for criterion in CRITERIA:
                        results = accumulate_prompts(scenario, criterion, cf_content, sum_par_content, None, cf_filename, client)
                        save_results(results, scenario, criterion, cf_stem, paragraph_index=i)
            else:
                for criterion in CRITERIA:
                    results = accumulate_prompts(scenario, criterion, cf_content, summary_content, None, cf_filename, client)
                    save_results(results, scenario, criterion, cf_stem)


if __name__ == '__main__':
    BASE_DIR = pathlib.Path(__file__).parent.parent
    with open(BASE_DIR / "config.json") as f:
        CONFIG = json.load(f)

    print("What would you like to do?")
    print("1. Preprocess only")
    print("2. Generate prompts only")
    print("3. Run full pipeline (preprocess + generate)")
    print("4. Encode results to CSV")
    print("5. Run everything")
    choice = CONFIG["task"]

    if choice in ("1", "2", "3", "5"):
        print("Which model?")
        print("1. Llama 8B (local GPU)")
        print("2. Groq (Llama 8B via API)")
        print("3. Claude API")
        model_choice = input("Enter 1, 2, or 3: ").strip()
        if model_choice == "1":
            device = int(os.environ.get("DEVICE", 1))
            client = hf_pipeline(
                "text-generation",
                model=CONFIG["sum_model_path"],
                device=device
            )
        elif model_choice == "2":
            client = api_models.GroqClient()
        else:
            client = api_models.ClaudeClient()

    if choice in ("1", "3", "5"):
        logger.info("=== Preprocessing ===")
        run_preprocess(client)
    if choice in ("2", "3", "5"):
        logger.info("=== Generating Adversarial Prompts ===")
        run_autoprompt(client)
    if choice in ("4", "5"):
        pathdir = input("Please enter the directory path: \n")
        encoder_(pathdir)