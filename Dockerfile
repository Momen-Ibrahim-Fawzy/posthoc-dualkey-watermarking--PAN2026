# ── PAN@CLEF 2026 Text Watermarking — GPU/BERT-LS Dockerfile ──────────────────
#
# Build:  docker build -t pan26-watermark .
#
# Usage:
#   Watermark:  python main.py watermark <input_dir> <output_dir>
#   Detect:     python main.py detect    <input_dir> <output_dir>
#
# Note: all NLTK data and BERT models are pre-downloaded during docker build
# so the container can run in offline environments.
# ──────────────────────────────────────────────────────────────────────────────

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8

# ── System packages ────────────────────────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ── Model cache directory ──────────────────────────────────────────────────────
ENV TRANSFORMERS_CACHE=/models
ENV HF_HOME=/models
ENV SENTENCE_TRANSFORMERS_HOME=/models/sentence-transformers

# ── Base Python packages ───────────────────────────────────────────────────────
RUN pip3 install click pandas nltk tqdm

# ── CPU-only PyTorch (separate install — needs its own --index-url) ───────────
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# ── BERT packages ──────────────────────────────────────────────────────────────
RUN pip3 install transformers sentence-transformers

# ── Pre-download NLTK corpora (required at runtime, no internet) ───────────────
ADD setup_nltk.py /
RUN python3 /setup_nltk.py

# ── Pre-download BERT / SBERT models into /models ─────────────────────────────
ADD bert_synonyms.py setup_models.py /
RUN python3 /setup_models.py

# ── Offline mode ───────────────────────────────────────────────────────────────
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# ── Application source ─────────────────────────────────────────────────────────
ADD config.py synonyms.py watermarker.py detector.py main.py /

RUN chmod +x /main.py
