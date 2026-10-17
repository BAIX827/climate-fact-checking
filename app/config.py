"""Service settings, overridable via environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _path(name, default):
    return Path(os.getenv(name, str(default)))


EVIDENCE_PATH = _path("EVIDENCE_PATH", ROOT / "data" / "evidence.json")
SAMPLE_PATH = ROOT / "data" / "sample_evidence.json"
INDEX_DIR = _path("INDEX_DIR", ROOT / "index_cache")
MODEL_DIR = _path("MODEL_DIR", ROOT / "models")  # optional fine-tuned cross_encoder.pt

CROSS_MODEL = os.getenv("CROSS_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
NLI_MODEL = os.getenv("NLI_MODEL", "cross-encoder/nli-distilroberta-base")

BM25_POOL = int(os.getenv("BM25_POOL", 1000))  # BM25 candidates before keyword fusion
SHORTLIST = int(os.getenv("SHORTLIST", 100))  # passages sent to the cross-encoder
TOP_K = int(os.getenv("TOP_K", 4))  # evidence returned (k = 4 was best on dev)
ALPHA = float(os.getenv("ALPHA", 1.2))  # keyword-overlap weight tuned on dev
NLI_THRESHOLD = float(os.getenv("NLI_THRESHOLD", 0.5))  # min SUPPORTS/REFUTES prob


def evidence_path():
    """Full corpus if present, otherwise the bundled demo sample."""
    return EVIDENCE_PATH if EVIDENCE_PATH.exists() else SAMPLE_PATH
