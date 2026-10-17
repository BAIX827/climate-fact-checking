"""Repository paths, labels and JSON helpers shared by all stages."""

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("FACTCHECK_DATA_DIR", REPO / "data"))  # inputs, incl. evidence.json
WORK_DIR = Path(os.getenv("FACTCHECK_WORK_DIR", REPO))  # generated artefacts
PROC_DIR = WORK_DIR / "processed"  # stage 1 output
SL_DIR = WORK_DIR / "shortlists"  # stage 2 output
RERANK_DIR = WORK_DIR / "reranked"  # stage 3 output
MODEL_DIR = WORK_DIR / "models"  # stage 3-4 checkpoints
FIG_DIR = WORK_DIR / "outputs" / "figures"

LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
