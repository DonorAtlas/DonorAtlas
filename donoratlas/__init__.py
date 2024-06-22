import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATE_ABBREV_TO_NAME = json.load(
    open(os.path.join(BASE_DIR, "static", "states.json"), encoding="utf-8")
)
STATE_NAME_TO_ABBREV = {v: k for k, v in STATE_ABBREV_TO_NAME.items()}


__all__ = ["STATE_ABBREV_TO_NAME", "STATE_NAME_TO_ABBREV"]
