import pathlib
from pathlib import Path
curr_dir = pathlib.Path(__file__).parent.absolute()
CONTEXT_DIR = Path(curr_dir / "Context")
if CONTEXT_DIR.exists():
    for consent_form_name in CONTEXT_DIR.iterdir():
        CF_STEM = consent_form_name.stem
        print(CF_STEM)
        cf_dir = CONTEXT_DIR / CF_STEM