"""
utils.py — Shared utilities for AI-Hiring-Intelligence pipeline.

Provides:
    - Logging setup
    - Text tokenisation
    - Min-max normalisation
    - Safe text concatenation
    - Stopword set
    - Reproducibility seed setter
"""

import re
import math
import logging
import numpy as np
import pandas as pd
from typing import List, Optional

# ── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str = "ai_hiring", level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger.

    Args:
        name:  Logger name (default: "ai_hiring").
        level: Logging level (default: INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger()


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set NumPy random seed for deterministic outputs.

    Args:
        seed: Integer seed value (default: 42).
    """
    np.random.seed(seed)
    logger.info("Random seed set to %d", seed)


# ── Text utilities ────────────────────────────────────────────────────────────

STOPWORDS: set = set("""
a an the is are was were be been being have has had do does did will would could should
may might must can shall of in on at to for from with by about as into through during
before after above below between among through into out off over under then once this that
these those i me my we our you your he she it his her its they them their what which who
when where why how all each every both either neither one two three also just only very
most more than such any now use used using make made new well good high low work
working within across after before since
""".split())


def tokenize(text: str) -> List[str]:
    """Lowercase, clean, and tokenise text into word tokens.

    Args:
        text: Raw input string.

    Returns:
        List of lowercase alphanumeric tokens (length >= 3).
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return re.findall(r"\b[a-z][a-z0-9\-]{2,}\b", text)


def safe_concat(*parts: Optional[str]) -> str:
    """Safely concatenate multiple optional text fields into one lowercase string.

    Args:
        *parts: Variable number of string/None values.

    Returns:
        Single space-joined lowercase string.
    """
    cleaned = []
    for p in parts:
        if p is None:
            continue
        if isinstance(p, float) and math.isnan(p):
            continue
        cleaned.append(str(p).lower().strip())
    return " ".join(cleaned)


# ── Normalisation ─────────────────────────────────────────────────────────────

def minmax_norm(series: pd.Series) -> pd.Series:
    """Min-max normalise a pandas Series to [0, 1].

    Args:
        series: Numeric pandas Series.

    Returns:
        Normalised Series with same index. Returns zeros if all values equal.
    """
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def to_100(x: float) -> float:
    """Clip and round a normalised [0,1] float to a 0–100 score.

    Args:
        x: Float in approximately [0, 1].

    Returns:
        Float in [0.0, 100.0].
    """
    return float(np.clip(round(x * 100, 2), 0, 100))


# ── TF-IDF builder ────────────────────────────────────────────────────────────

def build_tfidf(docs: List[str], max_features: int = 8000):
    """Build a minimal TF-IDF vocabulary and IDF weights from a corpus.

    Args:
        docs:         List of raw text documents.
        max_features: Maximum vocabulary size (default: 8000).

    Returns:
        Tuple of (vocab: dict[str, int], idf: np.ndarray).
    """
    from collections import Counter
    df_counts: Counter = Counter()
    for doc in docs:
        tokens = set(tokenize(doc)) - STOPWORDS
        df_counts.update(tokens)

    N = len(docs)
    vocab_terms = [w for w, _ in df_counts.most_common(max_features)
                   if df_counts[w] >= 5]
    vocab = {w: i for i, w in enumerate(vocab_terms)}
    idf = np.array([
        math.log((N + 1) / (df_counts[w] + 1)) + 1
        for w in vocab_terms
    ], dtype=np.float32)
    logger.info("TF-IDF vocab built: %d terms from %d docs", len(vocab), N)
    return vocab, idf


def encode_text(text: str, vocab: dict, idf: np.ndarray) -> np.ndarray:
    """Encode a single document into a normalised TF-IDF vector.

    Args:
        text:  Raw text document.
        vocab: Vocabulary dict mapping token -> index.
        idf:   IDF weight array.

    Returns:
        L2-normalised float32 numpy array of length len(vocab).
    """
    from collections import Counter
    tokens = [t for t in tokenize(text) if t not in STOPWORDS]
    tf = Counter(tokens)
    total = max(1, len(tokens))
    vec = np.zeros(len(vocab), dtype=np.float32)
    for w, c in tf.items():
        if w in vocab:
            vec[vocab[w]] = (c / total) * idf[vocab[w]]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec
