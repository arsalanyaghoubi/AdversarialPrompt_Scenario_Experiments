import json
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent.parent
CONTEXT_DIR = BASE_DIR / "Context"

with open(BASE_DIR / "config.json") as f:
    _cfg = json.load(f)

# Core configuration
TASK = _cfg["task"]
MODEL = _cfg["model"]
ENCODER_DIR = pathlib.Path(_cfg["encoder_dir"]) if _cfg.get("encoder_dir") else None

# Word count limits
MIN_WORD_COUNT = _cfg["min_word_count"]
MAX_WORD_COUNT = _cfg["max_word_count"]

# Content & targets
N_PARAGRAPHS = _cfg["n_paragraphs"]
CF_TARGET = _cfg["targets"]["CF"]
SUM_TARGET = _cfg["targets"]["SUM"]
PAR_TARGET = _cfg["targets"]["PAR"]
SUM_PAR_TARGET = _cfg["targets"]["SUM_PAR"]

# File & model paths
CF_STEM = _cfg["cf_stem"]
SUM_MODEL_PATH = _cfg["sum_model_path"]
AP_MODEL_PATHS = _cfg["ap_model_paths"]

# Repetition penalty
REPETITION_PENALTY = _cfg["repetition_penalty"]