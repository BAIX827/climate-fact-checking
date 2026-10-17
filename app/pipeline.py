"""Inference pipeline: BM25 + keyword shortlist -> cross-encoder rerank -> NLI label."""

import hashlib
import json
import logging
import sys

import bm25s
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app import config

sys.path.insert(0, str(config.ROOT / "src"))
from factcheck.keywords import extract_keywords, fuse, keyword_overlap  # noqa: E402
from factcheck.text import clean_for_bert, clean_for_bm25, clean_for_keywords  # noqa: E402

log = logging.getLogger(__name__)
torch.set_grad_enabled(False)


class Retriever:
    """BM25 over the cleaned corpus, re-scored with keyword overlap (stage 2)."""

    def __init__(self, evidence: dict, index_dir=config.INDEX_DIR):
        self.ids = list(evidence)
        self.text = {i: clean_for_bert(t) for i, t in evidence.items()}
        self.kw = [clean_for_keywords(evidence[i]) for i in self.ids]
        digest = hashlib.md5("".join(self.ids).encode()).hexdigest()[:12]
        path = index_dir / f"bm25_{digest}"
        if path.exists():
            self.bm25 = bm25s.BM25.load(str(path))
            log.info("loaded BM25 index from %s", path)
        else:
            corpus = [clean_for_bm25(evidence[i]) for i in self.ids]
            self.bm25 = bm25s.BM25(k1=1.2, b=0.75)
            self.bm25.index(bm25s.tokenize(corpus, stopwords="en", stemmer=self._stemmer()))
            path.parent.mkdir(parents=True, exist_ok=True)
            self.bm25.save(str(path))
            log.info("built BM25 index (%d passages) -> %s", len(self.ids), path)

    @staticmethod
    def _stemmer():
        import Stemmer

        return Stemmer.Stemmer("english")

    def search(self, claim, pool=config.BM25_POOL, top_m=config.SHORTLIST, alpha=config.ALPHA):
        q = bm25s.tokenize(
            clean_for_bm25(claim), stopwords="en", stemmer=self._stemmer(), show_progress=False
        )
        if not q.vocab:
            return []
        docs, scores = self.bm25.retrieve(q, k=min(pool, len(self.ids)), show_progress=False)
        docs, scores = docs[0], scores[0]
        ents, kws = extract_keywords(claim)
        if kws:
            scores = fuse(scores, [keyword_overlap(self.kw[d], ents, kws) for d in docs], alpha)
        return [self.ids[docs[i]] for i in np.argsort(-scores)[:top_m]]


def _load(name):
    return AutoTokenizer.from_pretrained(name), AutoModelForSequenceClassification.from_pretrained(
        name
    ).eval()


class Reranker:
    """Cross-encoder relevance scores. Uses MODEL_DIR/cross_encoder.pt (from notebook 03) if present."""

    def __init__(self, name=config.CROSS_MODEL, model_dir=config.MODEL_DIR):
        self.tok, self.model = _load(name)
        ckpt = model_dir / "cross_encoder.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location="cpu")
            self.model.load_state_dict({k.removeprefix("backbone."): v for k, v in state.items()})
            log.info("loaded fine-tuned cross-encoder from %s", ckpt)

    def score(self, claim, passages, batch_size=32):
        out = []
        for i in range(0, len(passages), batch_size):
            chunk = passages[i : i + batch_size]
            enc = self.tok(
                [claim] * len(chunk),
                chunk,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            out.extend(self.model(**enc).logits.squeeze(-1).tolist())
        return out


class Labeller:
    """Zero-shot NLI per passage (premise = evidence, hypothesis = claim).

    The most decisive passage (highest entailment or contradiction probability)
    sets the label if that probability >= NLI_THRESHOLD, else NOT_ENOUGH_INFO.
    DISPUTED is not predicted; it needs the fine-tuned classifier from notebook 04.
    """

    MAP = {"entailment": "SUPPORTS", "contradiction": "REFUTES", "neutral": "NOT_ENOUGH_INFO"}

    def __init__(self, name=config.NLI_MODEL, threshold=config.NLI_THRESHOLD):
        self.tok, self.model = _load(name)
        self.labels = [self.MAP[l.lower()] for _, l in sorted(self.model.config.id2label.items())]
        self.threshold = threshold

    def predict(self, claim, passages):
        if not passages:
            return "NOT_ENOUGH_INFO", {}
        enc = self.tok(
            passages,
            [claim] * len(passages),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        probs = self.model(**enc).logits.softmax(-1)
        best = {l: probs[:, i].max().item() for i, l in enumerate(self.labels)}
        label = max(("SUPPORTS", "REFUTES"), key=best.get)
        return (label if best[label] >= self.threshold else "NOT_ENOUGH_INFO"), best


class Pipeline:
    def __init__(self, evidence: dict, retriever=None, reranker=None, labeller=None):
        self.retriever = retriever or Retriever(evidence)
        self.reranker = reranker or Reranker()
        self.labeller = labeller or Labeller()

    @classmethod
    def from_config(cls):
        path = config.evidence_path()
        log.info("loading evidence from %s", path)
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def verify(self, claim, top_k=config.TOP_K):
        ids = self.retriever.search(claim)
        texts = [self.retriever.text[i] for i in ids]
        scores = self.reranker.score(claim, texts) if ids else []
        ranked = sorted(zip(ids, texts, scores), key=lambda x: -x[2])[:top_k]
        label, probs = self.labeller.predict(claim, [t for _, t, _ in ranked])
        return {
            "claim": claim,
            "label": label,
            "label_scores": probs,
            "evidence": [{"id": i, "text": t, "score": round(s, 4)} for i, t, s in ranked],
        }
