"""
WordNet-based synonym lookup utilities with LRU caching.

We cache all synonym and similarity lookups to avoid redundant disk I/O across
the many repeated words in a corpus of political speeches.

Quality note
------------
WordNet synsets are ordered by frequency (most common sense first).  When we
look at ALL synsets for a word, we collect synonyms from rare/secondary senses
that are semantically distant in context (e.g. "global" → "globular" because
WordNet has a secondary sense of "global" meaning spherical).

To prevent such poor substitutions we do two things:
  1. Collect synonyms from at most the first PRIMARY_SENSES synsets.
  2. Require primary-sense path similarity ≥ MIN_SYNONYM_SIMILARITY before
     any synonym is accepted.  This blocks synonyms that share only an obscure
     secondary meaning with the original word.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()

# Number of most-frequent WordNet senses to scan, by POS.
#
# Adjectives (pos='a'): restricted to the primary synset (co-lemma synonyms only)
#   to avoid cross-sense accidents like "robust"→"racy".
# Nouns (pos='n'): top 3 senses give a good pool; similarity filter maintains quality.
# Verbs (pos='v'): top 2 senses; verb hierarchy is fine-grained so 2 is sufficient.
PRIMARY_SENSES_BY_POS: dict[str, int] = {"a": 1, "n": 3, "v": 2}

# Minimum primary-sense path similarity required to accept a synonym, by POS.
# Adjectives: controlled by synset-depth limit above (no similarity check needed).
# Nouns:      0.18 — loose enough for political vocabulary, tight enough to block
#             "solution"/"solvent" cross-sense accidents at lower values.
# Verbs:      0.34 — just above 1/3 to exclude the common 0.333 path-similarity
#             co-lemma accidents (e.g. find→happen, find→chance) while keeping
#             true co-lemma synonyms that share the same primary synset (sim=1.0)
#             such as consider→regard, require→necessitate, ensure→assure.
MIN_SYNONYM_SIMILARITY = 0.18   # nouns (kept for backward-compat reference)
_MIN_SIM_BY_POS: dict[str, float] = {"n": 0.18, "v": 0.34, "a": 0.0}


# ── Core utilities ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=50_000)
def get_lemma(word: str, wn_pos: str) -> str:
    """Return the WordNet lemma (base form) for *word*."""
    return _lemmatizer.lemmatize(word.lower(), pos=wn_pos)


@lru_cache(maxsize=100_000)
def primary_similarity(word: str, synonym: str, wn_pos: str) -> float:
    """Path similarity between PRIMARY (most-frequent) synsets only.

    Comparing first synsets prevents high similarity scores that arise only
    from shared secondary/rare meanings between two otherwise different words.
    """
    w_synsets = wn.synsets(get_lemma(word, wn_pos), pos=wn_pos)
    s_synsets = wn.synsets(synonym, pos=wn_pos)
    if not w_synsets or not s_synsets:
        return 0.0
    sim = wn.path_similarity(w_synsets[0], s_synsets[0])
    return sim if sim is not None else 0.0


@lru_cache(maxsize=50_000)
def get_synonyms(word: str, wn_pos: str) -> tuple[str, ...]:
    """Return a sorted tuple of quality-filtered WordNet synonyms for *word*.

    Strategy differs by POS:

    Adjectives (wn_pos='a'):
      Only accept lemmas that share the word's PRIMARY synset (synset depth=1).
      This gives genuine co-lemma synonyms ("revolutionary"↔"radical",
      "effective"↔"efficacious") and automatically blocks cross-sense accidents
      ("robust"↔"racy", which share only the secondary wine-tasting synset).
      No additional similarity filter is needed because co-lemma synonyms in the
      same synset are by definition semantically equivalent.

    Nouns (wn_pos='n'):
      Scan the top PRIMARY_SENSES_BY_POS['n'] synsets for a wider synonym pool,
      then apply the path-similarity gate (MIN_SYNONYM_SIMILARITY) against the
      word's primary synset to block distant-sense synonyms.

    Common filters (both POS):
      - Single tokens only (no underscore multi-word expressions)
      - Alphabetic, minimum length 4 characters
      - Must differ from the original word (case-insensitive) and its lemma
    """
    lemma = get_lemma(word, wn_pos)
    depth = PRIMARY_SENSES_BY_POS.get(wn_pos, 3)
    synonyms: set[str] = set()

    for synset in wn.synsets(lemma, pos=wn_pos)[:depth]:
        for lemma_obj in synset.lemmas():
            syn = lemma_obj.name()
            if (
                "_" not in syn
                and syn.isalpha()
                and len(syn) >= 4
                and syn.lower() != word.lower()
                and syn.lower() != lemma
            ):
                # For nouns and verbs: apply POS-specific primary-sense similarity gate.
                # For adjectives: the synset-depth limit is sufficient.
                min_sim = _MIN_SIM_BY_POS.get(wn_pos, MIN_SYNONYM_SIMILARITY)
                if min_sim > 0.0:
                    if primary_similarity(word.lower(), syn, wn_pos) < min_sim:
                        continue
                synonyms.add(syn.lower())

    return tuple(sorted(synonyms))


def get_best_synonym(word: str, wn_pos: str, candidates: List[str]) -> Optional[str]:
    """Return the candidate with the highest primary-sense similarity to *word*.

    Returns None if no candidate passes the similarity threshold (shouldn't
    happen since get_synonyms already filtered, but defensive).
    Falls back to the first candidate when no similarity is computable.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    best_syn = candidates[0]
    best_sim = -1.0

    for syn in candidates[:12]:          # Cap at 12 to bound runtime
        sim = primary_similarity(word.lower(), syn, wn_pos)
        if sim > best_sim:
            best_sim = sim
            best_syn = syn

    return best_syn
