import os, json, pathlib, logging
from transformers import pipeline as hf_pipeline
from pipeline.preprocess import run_preprocess
from pipeline.prompt_gen import run_autoprompt
from clients import ClaudeClient, GroqClient
from utils.log_config import setup_logging
from utils.csv_encoder import encoder_

setup_logging()
logger = logging.getLogger(__name__)

TASK_RUNS_PREPROCESS  = {"preprocess", "full_pipeline", "full_pipeline_and_csv_encoder"}
TASK_RUNS_AUTOPROMPT  = {"prompt_generation", "full_pipeline", "full_pipeline_and_csv_encoder"}
TASK_RUNS_ENCODER     = {"csv_encoder", "full_pipeline_and_csv_encoder"}
TASK_NEEDS_CLIENT     = TASK_RUNS_PREPROCESS | TASK_RUNS_AUTOPROMPT

MODEL_CHOICES = {"local", "groq", "claude"}

if __name__ == '__main__':
    BASE_DIR = pathlib.Path(__file__).parent
    with open(BASE_DIR / "config.json") as f:
        CONFIG = json.load(f)

    task  = CONFIG["task"]
    model = CONFIG["model"]

    if task not in TASK_RUNS_PREPROCESS | TASK_RUNS_AUTOPROMPT | TASK_RUNS_ENCODER:
        raise ValueError(f"Unknown task: '{task}'. Check config.json.")
    if model not in MODEL_CHOICES:
        raise ValueError(f"Unknown model: '{model}'. Must be one of {MODEL_CHOICES}.")

    if task in TASK_NEEDS_CLIENT:
        if model == "local":
            device = int(os.environ.get("DEVICE", 1))
            client = hf_pipeline(
                "text-generation",
                model=CONFIG["sum_model_path"],
                device=device
            )
        elif model == "groq":
            client = GroqClient()
        else:
            client = ClaudeClient()
        logger.info("Using model: %s", model)

    if task in TASK_RUNS_PREPROCESS:
        logger.info("=== Preprocessing ===")
        run_preprocess(client)
    if task in TASK_RUNS_AUTOPROMPT:
        logger.info("=== Generating Adversarial Prompts ===")
        run_autoprompt(client)
    if task in TASK_RUNS_ENCODER:
        logger.info("=== Encoding Results to CSV ===")
        encoder_(CONFIG["encoder_dir"])


        # some tasks exist under two or more scenarios : full_pipeline_and_csv_encoder
        # Branch naming
        # Model Choice: Llama ??!!
        # Task: prompt generation