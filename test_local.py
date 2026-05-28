#!/usr/bin/env python3
"""Quick local smoke-test for the watermarking system."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import compute_green_bit, TARGET_POS_MAP
from synonyms import get_synonyms
from watermarker import TextWatermarker
from detector import WatermarkDetector
import nltk

TEXT = (
    "Mr President, the digital services act is a landmark legislation with the potential "
    "to become a global standard. While it is not perfect, it is still revolutionary. "
    "Why is that so? First of all, it breaks with the paradigm of inevitability, and it "
    "puts democracy over private, monopolistic interests, because until now there was a "
    "feeling that big tech was too powerful to regulate. We need to ensure that the "
    "implementation is effective and that the enforcement is robust and transparent. "
    "This is a significant challenge for our democratic institutions."
)

print("=== WATERMARKING TEST ===")
wm = TextWatermarker()
watermarked = wm.watermark(TEXT)

orig_words = TEXT.split()
water_words = watermarked.split()
diffs = [(o, w) for o, w in zip(orig_words, water_words) if o.lower() != w.lower()]
print(f"Original:   {TEXT[:180]}")
print(f"Watermarked:{watermarked[:180]}")
print(f"Changed words ({len(diffs)}): {diffs}")
print()

print("=== DETECTION TEST ===")
detector = WatermarkDetector()

z_w = detector.compute_z_score(watermarked)
z_o = detector.compute_z_score(TEXT)
lbl_w = detector.detect(watermarked)
lbl_o = detector.detect(TEXT)

print(f"Z watermarked={z_w:.3f}  label={lbl_w}  (expect 1.0)")
print(f"Z original   ={z_o:.3f}  label={lbl_o}  (expect 0.0)")

# Sentence-shuffle robustness
sentences = nltk.sent_tokenize(watermarked)
shuffled = " ".join(reversed(sentences))
z_sh = detector.compute_z_score(shuffled)
lbl_sh = detector.detect(shuffled)
print(f"Z shuffled   ={z_sh:.3f}  label={lbl_sh}  (expect 1.0, shuffle-robust)")
print()

print("=== GREEN WORD BREAKDOWN ===")
words = nltk.word_tokenize(watermarked)
pos_tags = nltk.pos_tag(words)
green_count = total_count = 0
for word, pos in pos_tags:
    if pos not in TARGET_POS_MAP or not word.isalpha() or len(word) < 4 or word.isupper():
        continue
    wn_pos = TARGET_POS_MAP[pos]
    syns = get_synonyms(word.lower(), wn_pos)
    if not syns:
        continue
    total_count += 1
    bit = compute_green_bit(word.lower())
    if bit == 1:
        green_count += 1
    print(f"  {word:20s} bit={bit}  syns={list(syns[:3])}")

if total_count > 0:
    print(f"\nGreen fraction: {green_count}/{total_count} = {green_count/total_count:.3f}")
