"""Retriever on the synthetic demo corpus (BM25 + keyword fusion, no neural models)."""

import json

from app import config
from app.pipeline import Retriever


def test_retriever_ranks_relevant_passage_first(tmp_path):
    evidence = json.loads(config.SAMPLE_PATH.read_text())
    r = Retriever(evidence, index_dir=tmp_path)
    assert r.search("Arctic sea ice is declining")[0] == "demo-03"
    assert Retriever(evidence, index_dir=tmp_path).search("ocean acidity")[0] == "demo-07"  # cached index
