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