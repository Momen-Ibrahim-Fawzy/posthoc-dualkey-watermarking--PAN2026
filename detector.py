"""
Watermark detection via a one-sided Z-test on green word fraction.

Algorithm
---------
For every word in the (potentially attacked) text:
  1. Normalise the text aggressively: NFKC unicode, homoglyph mapping,
     zero-width character removal, ligature expansion.  This defends against
     character-level perturbation attacks (homoglyph substitution, zero-width
     space insertion, etc.) that would otherwise cause scoreable words to be
     missed by the tokeniser.
  2. Keep only words in TARGET_POS_MAP (NN, JJ), alphabetic, len >= 4,
     not ALL-CAPS (acronyms).
  3. Require the word to have at least one WordNet synonym — these are the
     positions where watermark bits could have been embedded.  Words without
     synonyms contribute no information.
  4. Compute compute_green_bit(word.lower()) -> 0 or 1.
  5. Aggregate: green_count / total_count (type-level, each word counted once).

Z-test (one-sided):
  Under H0 (not watermarked): green fraction ~ Binomial(n, p0) / n
    -> mean p0, std sqrt(p0*(1-p0) / n)
  Under H1 (watermarked): green fraction well above p0 -> Z >> 0

  Z = (observed_fraction - p0) / sqrt(0.25 / n)

  Predict "watermarked" if Z >= Z_THRESHOLD.

Robustness properties
---------------------
* Sentence-order invariant: green/red bit depends only on the word, not
  its position -> sentence shuffle attack has zero effect.
* Partial-attack tolerant: each surviving green word still contributes
  evidence; the Z-score degrades gracefully as more words are attacked.
* Character-attack resistant: aggressive text normalisation unmasks
  homoglyph substitution and zero-width insertion before tokenisation.
* False-positive rate: ~6.7% per key (single key); ~0.45% for dual-key
  AND detection (both Z-tests must pass simultaneously).
"""

from __future__ import annotations

import math
import unicodedata

import nltk
from nltk.tokenize import word_tokenize

from config import (
    TARGET_POS_MAP, Z_THRESHOLD, MIN_SCOREABLE_WORDS,
    P0_GREEN, P0_GREEN_2, compute_green_bit, compute_green_bit_2,
)
from synonyms import get_synonyms

# ── Character-level attack defence ────────────────────────────────────────────
# Mapping of visually similar non-ASCII characters to their ASCII equivalents.
# Attackers may substitute these to disrupt tokenisation without changing the
# apparent text (e.g. replacing Latin 'a' with Cyrillic 'а').
_HOMOGLYPH_TABLE = str.maketrans({
    # Cyrillic letters that look like Latin
    'а': 'a',   # а → a
    'е': 'e',   # е → e
    'о': 'o',   # о → o
    'р': 'p',   # р → p
    'с': 'c',   # с → c
    'х': 'x',   # х → x
    'і': 'i',   # і → i
    'А': 'A',   # А → A
    'В': 'B',   # В → B
    'Е': 'E',   # Е → E
    'К': 'K',   # К → K
    'М': 'M',   # М → M
    'Н': 'H',   # Н → H
    'О': 'O',   # О → O
    'Р': 'P',   # Р → P
    'С': 'C',   # С → C
    'Т': 'T',   # Т → T
    'Х': 'X',   # Х → X
    # Greek letters that look like Latin
    'α': 'a',   # α → a
    'β': 'b',   # β → b
    'ο': 'o',   # ο → o
    'τ': 't',   # τ → t
    'Α': 'A',   # Α → A
    'Β': 'B',   # Β → B
    'Ε': 'E',   # Ε → E
    'Ζ': 'Z',   # Ζ → Z
    'Η': 'H',   # Η → H
    'Ι': 'I',   # Ι → I
    'Κ': 'K',   # Κ → K
    'Μ': 'M',   # Μ → M
    'Ν': 'N',   # Ν → N
    'Ο': 'O',   # Ο → O
    'Ρ': 'P',   # Ρ → P
    'Τ': 'T',   # Τ → T
    'Χ': 'X',   # Χ → X
    # Zero-width and invisible characters (strip by mapping to empty)
    '​': '',    # ZERO WIDTH SPACE
    '‌': '',    # ZERO WIDTH NON-JOINER
    '‍': '',    # ZERO WIDTH JOINER
    '­': '',    # SOFT HYPHEN
    '⁠': '',    # WORD JOINER
    '﻿': '',    # BOM (when not at start)
    '͏': '',    # COMBINING GRAPHEME JOINER
    # Ligatures (NFKC handles most, but cover remaining)
    'ﬀ': 'ff',  # ﬀ → ff
    'ﬁ': 'fi',  # ﬁ → fi
    'ﬂ': 'fl',  # ﬂ → fl
    'ﬃ': 'ffi', # ﬃ → ffi
    'ﬄ': 'ffl', # ﬄ → ffl
    'ﬅ': 'st',  # ﬅ → st
    'ﬆ': 'st',  # ﬆ → st
})


def _harden_text(text: str) -> str:
    """Normalise text to resist character-level perturbation attacks.

    Steps:
    1. NFKC unicode normalisation (handles most compatibility substitutions)
    2. Homoglyph mapping (Cyrillic/Greek look-alikes → ASCII)
    3. Strip any remaining zero-width / invisible characters
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_HOMOGLYPH_TABLE)
    # Strip any remaining non-printable control characters (except whitespace)
    text = "".join(ch for ch in text if ch.isprintable() or ch in " \t\n\r")
    return text


class WatermarkDetector:
    """Detect whether a text carries the green-list synonym watermark."""

    def __init__(self, threshold: float = Z_THRESHOLD) -> None:
        self.threshold = threshold

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, text: str) -> float:
        """Return 1.0 if BOTH key-1 and key-2 Z-tests pass, else 0.0.

        Requiring two independent Z-tests reduces P(FP) from ~6.7% to ~0.45%,
        cutting expected false positives on a 30-text sample from ~2 to ~0.1.
        """
        z1 = self.compute_z_score(text, compute_green_bit, P0_GREEN)
        if z1 < self.threshold:
            return 0.0  # fast path: fail key-1
        z2 = self.compute_z_score(text, compute_green_bit_2, P0_GREEN_2)
        return 1.0 if z2 >= self.threshold else 0.0

    def compute_z_score(self, text: str, green_bit_fn=None, p0: float = P0_GREEN) -> float:
        """Compute the watermark Z-score for *text* using *green_bit_fn*.

        Defaults to key-1 (compute_green_bit) when green_bit_fn is None.
        Returns 0.0 when fewer than MIN_SCOREABLE_WORDS unique scoreable types
        are found (insufficient evidence).
        """
        if green_bit_fn is None:
            green_bit_fn = compute_green_bit

        text = _harden_text(text)
        words = word_tokenize(text)
        pos_tags = nltk.pos_tag(words)

        green_count = 0
        total_count = 0
        seen_types: set[str] = set()

        for word, pos in pos_tags:
            if pos not in TARGET_POS_MAP:
                continue
            if not word.isalpha() or len(word) < 4 or word.isupper():
                continue

            word_lower = word.lower()
            if word_lower in seen_types:
                continue
            seen_types.add(word_lower)

            wn_pos = TARGET_POS_MAP[pos]

            if not get_synonyms(word_lower, wn_pos):
                continue

            total_count += 1
            if green_bit_fn(word_lower) == 1:
                green_count += 1

        if total_count < MIN_SCOREABLE_WORDS:
            return 0.0

        observed_fraction = green_count / total_count
        z = (observed_fraction - p0) / math.sqrt(0.25 / total_count)
        return z
