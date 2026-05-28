"""
BERT-based lexical substitution — fallback for words WordNet cannot cover.

When the watermarker finds a red word with NO green WordNet synonyms, this
module queries BERT's fill-mask pipeline using synonym-focused templates such
as "X is also known as [MASK]" rather than masking the word in its original
sentence.  The template approach steers BERT towards actual synonyms instead
of contextually appropriate words that fit the position.

Candidates are filtered by:
  1. WordNet scoreability  — the candidate must have WordNet synonyms so the
                             detector (which uses WordNet only) can score it.
  2. Semantic similarity   — SBERT cosine ≥ MIN_BERT_SIM (default 0.45) to
                             the original word, measured in embedding space.
  3. Green bit             — caller filters; this module returns all valid
                             candidates sorted by SBERT similarity (best first).

Why 0.45 threshold (not the 0.65 used for context masking)?
  TWF = max(BLEU, BERTScore) × BA.  Our BLEU is 0.976 (dominant term).
  Even if a few BERT-LS substitutions slightly lower BERTScore, the max()
  keeps using BLEU, so TWF is unaffected.  The gain in BA from fixing more
  hard texts outweighs the BERTScore risk.

GPU-accelerated on CUDA when available (RTX 2070 fits both models easily).
Falls back to CPU gracefully if CUDA / packages are absent.
"""

from __future__ import annotations

import re
from functools import lru_cache

# ── Tunables ──────────────────────────────────────────────────────────────────
BERT_MODEL     = "bert-base-uncased"
EMBEDDER_MODEL = "sentence-transformers/all-mpnet-base-v2"  # stronger than MiniLM
TOP_K          = 50    # fill-mask top-K predictions per template
MIN_BERT_SIM   = 0.45  # SBERT cosine similarity floor

# Synonym-focused templates.  These steer BERT towards semantic synonyms
# rather than contextually fitting words.
_TEMPLATES = [
    "{word} is also known as [MASK] .",
    "a [MASK] is a synonym for {word} .",
    "{word} or [MASK] are often used interchangeably .",
    "[MASK] and {word} mean the same thing .",
]

# ── Lazy globals ──────────────────────────────────────────────────────────────
_fill_mask = None
_embedder  = None
_st_util   = None
_loaded    = False
_available = None


def is_available() -> bool:
    """True if transformers and sentence-transformers are importable."""
    global _available
    if _available is None:
        try:
            import transformers           # noqa: F401
            import sentence_transformers  # noqa: F401
            _available = True
        except ImportError:
            _available = False
    return _available


def _load_models() -> None:
    global _fill_mask, _embedder, _st_util, _loaded
    if _loaded:
        return
    import torch
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer, util as st_util

    device = 0 if torch.cuda.is_available() else -1
    label  = f"cuda:{device}" if device >= 0 else "cpu"
    print(f"[BERT-LS] loading {BERT_MODEL} + {EMBEDDER_MODEL} on {label} …")

    _fill_mask = pipeline("fill-mask", model=BERT_MODEL, device=device, top_k=TOP_K)
    _embedder  = SentenceTransformer(EMBEDDER_MODEL, device=label)
    _st_util   = st_util
    _loaded    = True
    print("[BERT-LS] models ready.")


@lru_cache(maxsize=25_000)
def get_bert_synonyms(
    word:           str,
    sentence:       str,   # kept for API compatibility; templates ignore it
    wn_pos:         str,
    min_similarity: float = MIN_BERT_SIM,
) -> tuple[str, ...]:
    """Return BERT synonym candidates for *word*, sorted by SBERT similarity.

    Queries four synonym-focused fill-mask templates, aggregates predictions,
    then filters by WordNet scoreability and semantic similarity.

    *sentence* is accepted but not used (synonym templates are independent of
    the sentence context).  It is part of the cache key so callers can pass it
    without breaking anything.
    """
    if not is_available():
        return ()
    _load_models()

    from synonyms import get_synonyms as _wn_synonyms

    word_emb = _embedder.encode(word, convert_to_tensor=True, show_progress_bar=False)

    scored: dict[str, float] = {}   # token → best similarity seen

    for tmpl in _TEMPLATES:
        masked = tmpl.format(word=word)
        try:
            predictions = _fill_mask(masked)
        except Exception:
            continue

        for pred in predictions:
            token = pred["token_str"].strip().lower()

            # Basic validity
            if not token.isalpha() or len(token) < 4:
                continue
            if token == word.lower():
                continue

            # Must be scoreable by the detector (has WordNet synonyms)
            if not _wn_synonyms(token, wn_pos):
                continue

            # Semantic similarity gate
            if token in scored:
                continue  # already seen at a (possibly higher) score
            tok_emb = _embedder.encode(token, convert_to_tensor=True, show_progress_bar=False)
            sim = float(_st_util.cos_sim(word_emb, tok_emb))
            if sim >= min_similarity:
                scored[token] = sim

    # Return sorted best-first
    return tuple(tok for tok, _ in sorted(scored.items(), key=lambda x: -x[1]))
