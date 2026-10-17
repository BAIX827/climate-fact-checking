"""Three cleaned views of each text: BERT (light), BM25 (aggressive + abbreviation
expansion) and keyword (lowercase alphanumeric)."""

import re
import string
import unicodedata

from unidecode import unidecode

# Domain-specific replacements — APPLIED ONLY TO THE BM25 VIEW.
# These help sparse retrieval because the BM25 index won't otherwise tie
# 'CO2' to 'carbon dioxide' even though they refer to the same entity.
SPECIAL_TOKEN_MAP = [
    (r"\bCO\s*2\b", " co2 carbon dioxide "),  # keep BOTH forms
    (r"CO\u2082", " co2 carbon dioxide "),
    (r"\bH2O\b", " h2o water "),
    (r"\bCH\s*4\b", " ch4 methane "),
    (r"\bN2O\b", " n2o nitrous oxide "),
    (r"\bNO\s*x\b", " nox nitrogen oxides "),
    (r"\bSO\s*2\b", " so2 sulfur dioxide "),
    (r"\bO\s*3\b", " o3 ozone "),
    (r"°\s*C", " degrees celsius "),
    (r"°\s*F", " degrees fahrenheit "),
    (r"\bIPCC\b", " ipcc intergovernmental panel on climate change "),
    (r"\bGHG\b", " ghg greenhouse gas "),
    (r"\bAGW\b", " agw anthropogenic global warming "),
    (r"\bUN\b", " un united nations "),
    (r"\bUSA\b", " usa united states of america "),
    (r"\bUS\b", " us united states "),
    (r"\bUK\b", " uk united kingdom "),
    (r"\bEU\b", " eu european union "),
    (r"\bppm\b", " ppm parts per million "),
    (r"\bppb\b", " ppb parts per billion "),
]
_SPECIAL_PATTERNS = [(re.compile(p, flags=re.IGNORECASE), r) for p, r in SPECIAL_TOKEN_MAP]


def _replace_specials(s):
    for pat, rep in _SPECIAL_PATTERNS:
        s = pat.sub(rep, s)
    return s


# Unicode primitives
_WS_RE = re.compile(r"\s+")
_REPEAT_CHAR_RE = re.compile(r"([!?.,\-])\1{2,}")
_HYPHEN_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015]")
_QUOTE_RE = re.compile(r"[\u2018\u2019\u201A\u201B\u201C\u201D\u201E\u201F\u00AB\u00BB]")
_PUNCT_TRANS = str.maketrans({p: " " for p in string.punctuation})
_NON_ALNUM_TRANS = str.maketrans({c: " " for c in string.punctuation + "`~"})


def _strip_accents(s):
    return unidecode(unicodedata.normalize("NFC", s))


def _norm_quotes(s):
    return _QUOTE_RE.sub("'", _HYPHEN_RE.sub("-", s))


def _collapse_ws(s):
    return _WS_RE.sub(" ", s).strip()


def _collapse_rep(s):
    return _REPEAT_CHAR_RE.sub(r"\1", s)


def clean_for_bert(text):
    """Light normalisation only. Used for BERT/RoBERTa-based encoders & classifier."""
    if not isinstance(text, str):
        return ""
    t = _strip_accents(text)
    t = _norm_quotes(t)
    t = _collapse_rep(t)
    return _collapse_ws(t)


def clean_for_bm25(text):
    """Aggressive. Used to build the sparse BM25 index in stage 2."""
    if not isinstance(text, str):
        return ""
    t = _replace_specials(text)
    t = _strip_accents(t)
    t = _norm_quotes(t)
    t = _collapse_rep(t)
    t = t.lower()
    t = t.translate(_PUNCT_TRANS)
    return _collapse_ws(t)


def clean_for_keywords(text):
    """Lowercase + alphanumeric only. Used for the keyword-overlap signal
    so that 'IPCC' (a noun-chunk extracted from the claim) matches 'ipcc'
    inside the evidence even though we DON'T expand it on this view."""
    if not isinstance(text, str):
        return ""
    t = _strip_accents(text)
    t = t.lower()
    t = t.translate(_NON_ALNUM_TRANS)
    return _collapse_ws(t)
