# src/adversarial_pipeline/utils/logging.py
import logging

def setup_logging(level=logging.INFO, log_file="experiment.log"):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )