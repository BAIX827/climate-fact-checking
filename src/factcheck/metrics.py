"""Retrieval and end-to-end metrics (the latter mirrors the task scorer)."""

import numpy as np


def _f1(pred, gold):
    tp = len(set(pred) & set(gold))
    if not pred or tp == 0:
        return 0.0
    prec, rec = tp / len(set(pred)), tp / len(set(gold))
    return 2 * prec * rec / (prec + rec)


def recall_at(rankings, claims, k):
    """Micro recall of gold evidences within the top-k of each ranking."""
    hits = total = 0
    for cid, gold in claims.items():
        topk = set(rankings[cid][:k])
        total += len(gold["evidences"])
        hits += sum(ev in topk for ev in gold["evidences"])
    return hits / total if total else 0.0


def f1_at(rankings, claims, k=None):
    """Mean per-claim evidence F1 of the top-k (or full list if k is None)."""
    return float(np.mean([_f1(rankings[cid][:k], gold["evidences"]) for cid, gold in claims.items()]))


def harmonic_mean(a, b):
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def compute_eval_metrics(predictions, ground_truth):
    """Return (evidence F-score, label accuracy, harmonic mean)."""
    fs, accs = [], []
    for cid, gold in ground_truth.items():
        p = predictions.get(cid)
        if not p or "claim_label" not in p or "evidences" not in p:
            continue
        accs.append(float(p["claim_label"] == gold["claim_label"]))
        pred_ev = p["evidences"] if isinstance(p["evidences"], list) else []
        correct = sum(e in set(pred_ev) for e in gold["evidences"])
        if not pred_ev or correct == 0:
            fs.append(0.0)
            continue
        rec, prec = correct / len(gold["evidences"]), correct / len(pred_ev)
        fs.append(2 * prec * rec / (prec + rec))
    f, a = float(np.mean(fs)), float(np.mean(accs))
    return f, a, harmonic_mean(f, a)
