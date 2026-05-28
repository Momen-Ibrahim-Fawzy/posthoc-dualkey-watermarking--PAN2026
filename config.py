"""
Configuration for the PAN@CLEF 2026 Text Watermarking system.

Approach: Dual-key green-list synonym substitution (post-hoc KGW adaptation).
For each content word, HMAC(key, word) assigns it to the green or red partition.
During watermarking, red-listed word types are globally replaced with green-listed
synonyms. Detection uses a type-level Z-test on the green-type fraction.

Design properties:
  - Robust to sentence shuffling (hash is position-independent)
  - High-BLEU (synonym substitutions only, no restructuring)
  - High-BERTScore (synonyms preserve semantics)
  - Offline (no LLM, no internet required at inference time)
"""

import hmac
import hashlib
import os

# ── Secret watermarking keys ───────────────────────────────────────────────────
# Two independent keys embed two independent watermarks. Detection requires
# BOTH Z-tests to pass: P(FP) ≈ 0.067² ≈ 0.45% vs 6.7% for a single key.
#
# Set via environment variables for production deployments:
#   export WATERMARK_KEY_1="your-secret-key-1"
#   export WATERMARK_KEY_2="your-secret-key-2"
#
# The default values below are the keys used in the PAN 2026 competition
# submission. Change them for any new deployment — the keys determine the
# green/red partition, so watermarks created with different keys are not
# detectable by this detector.
SECRET_KEY   = os.getenv("WATERMARK_KEY_1",
               "pan26-syn-watermark-key1").encode()
SECRET_KEY_2 = os.getenv("WATERMARK_KEY_2",
               "pan26-syn-watermark-key2").encode()

# ── Detection threshold ────────────────────────────────────────────────────────
# Z-score threshold for the one-sided Z-test (both keys must exceed this).
# Z_THRESHOLD = 1.5 gives P(FP per key) ≈ 6.7%; dual-key P(FP) ≈ 0.45%.
Z_THRESHOLD: float = 1.5

# Embedding target: higher than detection threshold to maintain a safety margin
# that survives word-substitution attacks.
# Under a 30% word substitution attack, Z drops by ~30%:
#   Z_target = 2.5  →  post-attack Z ≈ 1.75  >>  Z_THRESHOLD  (safe)
Z_WATERMARK_TARGET: float = 2.5

# ── Empirical null calibration ─────────────────────────────────────────────────
# Natural green fraction of EU Parliament vocabulary under each key, measured
# on 300 training texts (avg 40 scoreable types each).
# Using calibrated p0 rather than 0.5 re-centres the Z-distribution for
# original texts, reducing false positives without hurting true positives.
P0_GREEN:   float = 0.513   # key 1 (measured: 6134/11950 green types)
P0_GREEN_2: float = 0.508   # key 2 (measured: 6068/11936 green types)

# ── Detection guard ────────────────────────────────────────────────────────────
# Minimum unique scoreable word types required to trust the Z-score.
MIN_SCOREABLE_WORDS: int = 4

# ── Watermarking parameters ────────────────────────────────────────────────────
# Maximum unique red word types to substitute per embedding stage.
MAX_TYPES_TO_SUBSTITUTE: int = 12

# ── Part-of-speech selection ───────────────────────────────────────────────────
# Maps NLTK POS tag → WordNet POS character.
# Plural nouns (NNS) are included because synonyms.py lemmatises them before
# WordNet lookup. Verb base/present forms match WordNet lemmas directly.
TARGET_POS_MAP: dict[str, str] = {
    "JJ":  "a",   # Adjective base form          → WordNet ADJ
    "NN":  "n",   # Noun singular/mass            → WordNet NOUN
    "NNS": "n",   # Noun plural (lemmatised)      → WordNet NOUN
    "VB":  "v",   # Verb base / infinitive form   → WordNet VERB
    "VBP": "v",   # Verb non-3rd person present   → WordNet VERB
}

# ── BERT-based lexical substitution (optional) ─────────────────────────────────
# When True, the watermarker queries BERT fill-mask as a fallback for red words
# that have NO green WordNet synonyms. Requires `transformers` and
# `sentence-transformers`. Set to False for a lightweight WordNet-only run.
USE_BERT_LS: bool = True


# ── HMAC helpers ───────────────────────────────────────────────────────────────

def _make_green_bit_fn(key: bytes):
    """Return a green-bit function bound to *key*."""
    def fn(word: str) -> int:
        h = hmac.new(key, word.lower().encode("utf-8"), hashlib.sha256)
        return int(h.hexdigest()[:8], 16) % 2
    return fn


compute_green_bit   = _make_green_bit_fn(SECRET_KEY)
compute_green_bit_2 = _make_green_bit_fn(SECRET_KEY_2)
