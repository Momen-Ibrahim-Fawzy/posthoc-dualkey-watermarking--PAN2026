"""
Text watermarking via dual-key green-list synonym substitution.

Two independent HMAC keys embed two independent watermarks; detection
requires BOTH Z-tests to pass → P(FP) ≈ 0.067² ≈ 0.45% vs 6.7% single-key.

Embedding strategy (three stages)
-----------------------------------
Stage 1 (Key-1): substitute RED-1 types with GREEN-1 synonyms until
                 Z1 ≥ Z_WATERMARK_TARGET.

Stage 2 (Key-2): substitute RED-2 types with GREEN-2 synonyms until
                 Z2 ≥ Z_WATERMARK_TARGET.  May incidentally lower Z1.

Stage 3 (Restore): if Stage 2 caused Z1 < Z_WATERMARK_TARGET, run another
                   Key-1 substitution round on any remaining RED-1 types
                   (including RED-1 synonyms introduced by Stage 2).

Within each stage, candidates are ranked frequency-DESCENDING within the
≥2-occurrence group (most-frequent types appear many times → attacker must
remove every occurrence → harder to erase under substitution attack).

Key-1 synonyms prefer DOUBLE-GREEN candidates (green for both keys) to
pre-boost Z2 "for free" and reduce how much work Stage 2 needs to do.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Callable, List, Optional, Set, Tuple

import nltk
from nltk.tokenize import word_tokenize

from config import (
    TARGET_POS_MAP, MAX_TYPES_TO_SUBSTITUTE, Z_THRESHOLD, Z_WATERMARK_TARGET,
    USE_BERT_LS, compute_green_bit, compute_green_bit_2,
)
from synonyms import get_synonyms, get_best_synonym

GreenBitFn = Callable[[str], int]


class TextWatermarker:
    """Embed dual-key green-list watermarks via global synonym substitution."""

    def __init__(self, max_types: int = MAX_TYPES_TO_SUBSTITUTE) -> None:
        self.max_types = max_types

    def watermark(self, text: str) -> str:
        """Return *text* with two independent watermarks embedded."""
        text = unicodedata.normalize("NFKC", text)

        from detector import WatermarkDetector
        det = WatermarkDetector()

        # Stage 1: embed Key-1 watermark; prefer double-green synonyms to
        # pre-boost Z2 for free and reduce Stage 2's workload.
        text, subs1 = self._watermark_pass(
            text, compute_green_bit, excluded=set(), prefer_double_green=True
        )

        # Stage 2: embed Key-2 watermark freely (may incidentally lower Z1).
        text, subs2 = self._watermark_pass(
            text, compute_green_bit_2, excluded=subs1, prefer_double_green=False
        )

        # Stage 3 (Dual-Key Restoration) — use a conservative target (just above
        # detection threshold) so neither key over-substitutes and exhausts vocabulary.
        # Full Z_WATERMARK_TARGET is for the initial stages (attack robustness).
        # Here we only need both Z-scores safely above Z_THRESHOLD.
        _CORR_TARGET = Z_THRESHOLD + 0.3   # 1.8 — detectable with comfortable margin

        for _ in range(3):
            z1 = det.compute_z_score(text, compute_green_bit)
            z2 = det.compute_z_score(text, compute_green_bit_2)
            if z1 >= Z_THRESHOLD and z2 >= Z_THRESHOLD:
                break
            excluded = subs1 | subs2
            if z1 < _CORR_TARGET:
                text, s = self._watermark_pass(
                    text, compute_green_bit, excluded=excluded,
                    prefer_double_green=False, target=_CORR_TARGET,
                )
                subs1 |= s
            if z2 < _CORR_TARGET:
                text, s = self._watermark_pass(
                    text, compute_green_bit_2, excluded=subs1 | subs2,
                    prefer_double_green=False, target=_CORR_TARGET,
                )
                subs2 |= s

        return text

    # ── Internal pass ──────────────────────────────────────────────────────────

    def _watermark_pass(
        self,
        text: str,
        green_bit_fn: GreenBitFn,
        excluded: Set[str],
        prefer_double_green: bool,
        target: Optional[float] = None,
    ) -> Tuple[str, Set[str]]:
        """One full watermark-embedding pass.

        green_bit_fn      — hash function for this key.
        excluded          — word types skipped (handled by an earlier stage).
        prefer_double_green — put double-green synonyms first in the list.
        target            — Z-score target for signal amplification (default: Z_WATERMARK_TARGET).

        Returns (watermarked_text, set_of_substituted_original_types).
        """
        if target is None:
            target = Z_WATERMARK_TARGET
        words    = word_tokenize(text)
        pos_tags = nltk.pos_tag(words)

        word_freq: Counter[str] = Counter()
        for word, pos in pos_tags:
            if pos not in TARGET_POS_MAP:
                continue
            if not word.isalpha() or len(word) < 4 or word.isupper():
                continue
            word_freq[word.lower()] += 1

        seen: set[str] = set()
        type_candidates: List[Tuple[str, str, int, tuple]] = []

        for word, pos in pos_tags:
            if pos not in TARGET_POS_MAP:
                continue
            if not word.isalpha() or len(word) < 4 or word.isupper():
                continue
            wl = word.lower()
            if wl in seen or wl in excluded:
                continue
            seen.add(wl)

            if green_bit_fn(wl) == 1:
                continue  # already green for this key

            wn_pos    = TARGET_POS_MAP[pos]
            wn_syns   = get_synonyms(wl, wn_pos)
            green_syns = _build_green_syns(wn_syns, green_bit_fn, prefer_double_green)
            if green_syns:
                type_candidates.append((wl, wn_pos, word_freq[wl], green_syns))

        # Rank: ≥2-occurrence group first, then frequency-DESCENDING within each
        # group.  Most-frequent types appear many times → each occurrence must be
        # individually removed by an attacker → more robust to substitution attacks.
        type_candidates.sort(key=lambda x: (0 if x[2] >= 2 else 1, -x[2]))
        selected      = type_candidates[: self.max_types]
        amplify_pool  = type_candidates[self.max_types :]

        # Primary substitution pass
        result         = text
        used_synonyms: set[str] = set()
        existing_types: set[str] = {w.lower() for w in words if w.isalpha()}
        substituted: set[str] = set()

        for word_lower, wn_pos, _, green_syns in selected:
            result, syn = _apply_substitution(
                word_lower, green_syns, result, used_synonyms, existing_types, wn_pos
            )
            if syn:
                used_synonyms.add(syn)
                substituted.add(word_lower)

        _det = None

        def _get_z() -> float:
            nonlocal _det
            if _det is None:
                from detector import WatermarkDetector
                _det = WatermarkDetector()
            return _det.compute_z_score(result, green_bit_fn)

        # Signal amplification (WordNet): draw from remaining candidates if below target
        if amplify_pool:
            z = _get_z()
            for word_lower, wn_pos, _, green_syns in amplify_pool:
                if z >= target:
                    break
                result, syn = _apply_substitution(
                    word_lower, green_syns, result, used_synonyms, existing_types, wn_pos
                )
                if syn:
                    used_synonyms.add(syn)
                    substituted.add(word_lower)
                    z = _get_z()

        # Signal amplification (BERT-LS): fallback for words WordNet cannot cover
        if USE_BERT_LS and _bert_ls_ready():
            z = _get_z()
            if z < target:
                result, bert_subs = _bert_ls_boost(
                    result, used_synonyms, existing_types,
                    target, _get_z, green_bit_fn, excluded,
                )
                substituted |= bert_subs

        return result, substituted


# ── Synonym helpers ────────────────────────────────────────────────────────────

def _build_green_syns(
    wn_syns: tuple,
    green_bit_fn: GreenBitFn,
    prefer_double_green: bool,
) -> tuple:
    """Return synonyms that are green for *green_bit_fn*.

    When *prefer_double_green* is True, synonyms that are also green for the
    OTHER key (compute_green_bit_2) are listed first.  This pre-boosts the
    secondary Z-score without extra substitutions.
    """
    if not prefer_double_green:
        return tuple(s for s in wn_syns if green_bit_fn(s) == 1)

    # Put double-green synonyms first
    double = tuple(
        s for s in wn_syns
        if green_bit_fn(s) == 1 and compute_green_bit_2(s) == 1
    )
    single = tuple(
        s for s in wn_syns
        if green_bit_fn(s) == 1 and compute_green_bit_2(s) == 0
    )
    return double + single


# ── BERT-LS boost ──────────────────────────────────────────────────────────────

def _bert_ls_boost(
    result:         str,
    used_synonyms:  set[str],
    existing_types: set[str],
    threshold:      float,
    get_z,
    green_bit_fn:   GreenBitFn,
    excluded:       Set[str],
) -> Tuple[str, Set[str]]:
    from bert_synonyms import get_bert_synonyms

    result_words = word_tokenize(result)
    result_pos   = nltk.pos_tag(result_words)
    result_sents = nltk.sent_tokenize(result)

    seen_bert: set[str] = set()
    substituted: set[str] = set()
    z = get_z()

    for word, pos in result_pos:
        if z >= threshold:
            break
        if pos not in TARGET_POS_MAP:
            continue
        if not word.isalpha() or len(word) < 4 or word.isupper():
            continue
        wl = word.lower()
        if wl in seen_bert or wl in excluded:
            continue
        seen_bert.add(wl)

        if green_bit_fn(wl) == 1:
            continue

        wn_pos  = TARGET_POS_MAP[pos]
        wn_syns = get_synonyms(wl, wn_pos)
        if any(green_bit_fn(s) == 1 for s in wn_syns):
            continue  # WordNet handles this in primary/amplification stages

        sentence   = _find_containing_sentence(wl, result_sents)
        bert_syns  = get_bert_synonyms(wl, sentence, wn_pos)
        green_bert = tuple(
            s for s in bert_syns
            if green_bit_fn(s) == 1 and s not in used_synonyms
        )
        if not green_bert:
            continue

        result, syn = _apply_substitution(
            wl, green_bert, result, used_synonyms, existing_types, wn_pos
        )
        if syn:
            used_synonyms.add(syn)
            substituted.add(wl)
            z = get_z()

    return result, substituted


# ── Module-level helpers ───────────────────────────────────────────────────────

_bert_ls_checked = False
_bert_ls_ok      = False


def _bert_ls_ready() -> bool:
    global _bert_ls_checked, _bert_ls_ok
    if not _bert_ls_checked:
        from bert_synonyms import is_available
        _bert_ls_ok      = is_available()
        _bert_ls_checked = True
        if not _bert_ls_ok:
            print("[watermarker] BERT-LS unavailable.")
    return _bert_ls_ok


def _find_containing_sentence(word: str, sentences: list[str]) -> str:
    pattern = r"\b" + re.escape(word) + r"\b"
    for sent in sentences:
        if re.search(pattern, sent, re.IGNORECASE):
            return sent
    return sentences[0] if sentences else word


def _apply_substitution(
    word_lower:     str,
    green_syns:     tuple,
    text:           str,
    used_synonyms:  set[str],
    existing_types: set[str],
    wn_pos:         str,
) -> tuple[str, str | None]:
    avail = [s for s in green_syns if s not in used_synonyms]
    if not avail:
        return text, None

    novel    = [s for s in avail if s not in existing_types]
    syn_pool = novel if novel else avail

    best_syn = get_best_synonym(word_lower, wn_pos, syn_pool)
    if best_syn is None:
        best_syn = syn_pool[0]

    pattern  = r"\b" + re.escape(word_lower) + r"\b"
    new_text = re.sub(
        pattern,
        lambda m, s=best_syn: _preserve_case(m.group(), s),
        text,
        flags=re.IGNORECASE,
    )
    return new_text, best_syn


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original and original[0].isupper():
        return replacement.capitalize()
    return replacement.lower()
