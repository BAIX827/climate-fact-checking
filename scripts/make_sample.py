"""Subset the full evidence.json: gold evidence for the given claims plus random distractors.

Usage: python scripts/make_sample.py --claims data/dev-claims.json --extra 5000 --out data/evidence_dev_subset.json
Then run the service with EVIDENCE_PATH=data/evidence_dev_subset.json.
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--evidence", default=ROOT / "data" / "evidence.json")
parser.add_argument("--claims", default=ROOT / "data" / "dev-claims.json")
parser.add_argument("--extra", type=int, default=5000, help="random non-gold passages to add")
parser.add_argument("--out", default=ROOT / "data" / "evidence_dev_subset.json")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

evidence = json.load(open(args.evidence, encoding="utf-8"))
claims = json.load(open(args.claims, encoding="utf-8"))
keep = {e for c in claims.values() for e in c.get("evidences", []) if e in evidence}
rest = [e for e in evidence if e not in keep]
keep |= set(random.Random(args.seed).sample(rest, min(args.extra, len(rest))))
json.dump({e: evidence[e] for e in sorted(keep)}, open(args.out, "w", encoding="utf-8"))
print(f"wrote {len(keep)} passages to {args.out}")
