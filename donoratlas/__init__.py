import json
import os

from donoratlas.names import PersonName, name_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

STATE_ABBREV_TO_NAME = json.load(open(os.path.join(PACKAGE_DIR, "static", "states.json"), encoding="utf-8"))
STATE_NAME_TO_ABBREV = {v: k for k, v in STATE_ABBREV_TO_NAME.items()}
STATE_CODES = STATE_ABBREV_TO_NAME.keys()


__all__ = ["STATE_ABBREV_TO_NAME", "STATE_NAME_TO_ABBREV", "STATE_CODES", "name_similarity", "PersonName"]
