# src/adversarial_pipeline/config.py
import json, pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent.parent
CONTEXT_DIR = BASE_DIR / "Context"

with open(BASE_DIR / "config.json") as f:
    _cfg = json.load(f)

TASK         = _cfg["task"]
N_PARAGRAPHS = _cfg["n_paragraphs"]
CF_TARGET    = _cfg["targets"]["CF"]
SUM_TARGET   = _cfg["targets"]["SUM"]
PAR_TARGET   = _cfg["targets"]["PAR"]
SUM_PAR_TARGET = _cfg["targets"]["SUM_PAR"]