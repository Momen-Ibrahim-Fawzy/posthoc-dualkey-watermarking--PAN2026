#!/usr/bin/env python3
"""Download all NLTK data required by the watermarking system.

This script is executed during `docker build` so that evaluation runs
(which have no internet access) find the data pre-installed in the image.
"""

import nltk

REQUIRED_PACKAGES = [
    "wordnet",                       # WordNet synonym database
    "omw-1.4",                       # Open Multilingual Wordnet (needed by wordnet)
    "punkt",                         # Sentence / word tokeniser
    "punkt_tab",                     # Punkt tokeniser (newer NLTK versions)
    "averaged_perceptron_tagger",    # POS tagger
    "averaged_perceptron_tagger_eng", # POS tagger — English (newer NLTK)
]

for pkg in REQUIRED_PACKAGES:
    print(f"Downloading: {pkg} …", flush=True)
    nltk.download(pkg, quiet=False)

print("\nAll NLTK data downloaded successfully.", flush=True)
