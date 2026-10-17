"""spaCy keyword extraction and BM25 + keyword-overlap score fusion (stage 2)."""

import re

import numpy as np

STOP_KW = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "it",
    "these",
    "those",
    "he",
    "she",
    "they",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
}
_KW_NORM_RE = re.compile(r"[^a-z0-9 ]+")
_nlp = None


def _norm_kw(s):
    return " ".join(_KW_NORM_RE.sub(" ", s.lower()).split())


def extract_keywords(text):
    """Return (entities, keywords); keywords = entities + noun chunks + content tokens."""
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    doc = _nlp(text)
    ents = {kw for kw in (_norm_kw(e.text) for e in doc.ents) if kw and len(kw) > 1 and kw not in STOP_KW}
    kws = set(ents)
    kws |= {
        kw for kw in (_norm_kw(c.text) for c in doc.noun_chunks) if kw and len(kw) > 2 and kw not in STOP_KW
    }
    kws |= {
        kw
        for kw in (
            _norm_kw(t.text) for t in doc if t.pos_ in ("NOUN", "PROPN", "NUM", "ADJ") and len(t.text) > 2
        )
        if kw and kw not in STOP_KW
    }
    return ents, kws


def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def keyword_overlap(ev_text_kw, ents, kws):
    """Weighted keyword hits (entities 2.0, others 1.0), normalised to ~[0, 1]."""
    if not kws:
        return 0.0
    ev = f" {ev_text_kw} "
    score = sum(2.0 if kw in ents else 1.0 for kw in kws if f" {kw} " in ev)
    return score / max(len(kws) + len(ents), 1)


def fuse(bm25_scores, overlaps, alpha):
    """Combined shortlist score: z(bm25) + alpha * keyword_overlap."""
    return zscore(bm25_scores) + alpha * np.asarray(overlaps, dtype=np.float32)
