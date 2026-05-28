#!/usr/bin/env python3
"""
Pre-download BERT and sentence-transformer models into the Docker image.
Run once during `docker build` so offline evaluation environments can find them.
"""
import os

# Point HuggingFace cache to the container's /models directory.
# Must be set BEFORE importing transformers/sentence_transformers.
os.environ.setdefault("TRANSFORMERS_CACHE", "/models")
os.environ.setdefault("HF_HOME", "/models")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/models/sentence-transformers")

from bert_synonyms import BERT_MODEL, EMBEDDER_MODEL

print(f"Downloading {BERT_MODEL} ...")
from transformers import AutoTokenizer, AutoModelForMaskedLM
AutoTokenizer.from_pretrained(BERT_MODEL)
AutoModelForMaskedLM.from_pretrained(BERT_MODEL)

print(f"Downloading {EMBEDDER_MODEL} ...")
from sentence_transformers import SentenceTransformer
SentenceTransformer(EMBEDDER_MODEL)

print("All models downloaded.")
