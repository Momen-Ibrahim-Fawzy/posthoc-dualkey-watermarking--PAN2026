# Post-Hoc Synonym Watermarking with Dual-Key Type-Level Detection

Official implementation of the system submitted to [PAN 2026 Text Watermarking](https://pan.webis.de/clef26/pan26-web/text-watermarking.html) task at CLEF 2026 — [`posthoc-dualkey-watermarking--PAN2026`](https://github.com/Momen-Ibrahim-Fawzy/posthoc-dualkey-watermarking--PAN2026).

> **Paper:** *Post-Hoc Synonym Watermarking with Dual-Key Type-Level Detection: A Precision--Robustness Trade-off at PAN 2026*
> Momen Ibrahim — Alexandria University
> CLEF 2026 Working Notes *(link will be added upon publication)*

## System Overview

The system embeds invisible watermarks into pre-existing human text by substituting red-listed word types with green-listed synonyms, then detects watermarks using a one-sided Z-test on the fraction of green word types.

| Component | Description |
|-----------|-------------|
| HMAC vocabulary partition | Deterministically assigns every word type to green/red using a secret key via HMAC-SHA256 |
| Type-level Z-test | Counts unique word types (not token occurrences) — better null distribution for repetitive political speech |
| WordNet synonym lookup | Primary synonym source with path-similarity filtering per POS |
| BERT-LS fallback | `bert-base-uncased` fill-mask with SBERT similarity gate for words lacking green WordNet synonyms |
| Dual-key AND detection | Two independent HMAC keys must both pass their Z-test → P(FP) ≈ 0.45% vs 6.7% single-key |

The central finding is a **precision–robustness trade-off**: the AND condition suppresses false positives to near zero but becomes brittle under attacks that independently suppress either key's signal.

**Official TIRA results:**

| System | Dataset | BA | TWF |
|--------|---------|-----|-----|
| **Dual-key** (this repo) | Spot-check | **0.995** | **0.971** |
| GPU v1 (single-key) | Official test | 0.904 | **0.775** |
| Dual-key (this repo) | Official test | 0.753 | 0.734 |

## Repository Structure

```
.
├── config.py          # All parameters and HMAC helpers
├── watermarker.py     # TextWatermarker — three-stage dual-key embedding
├── detector.py        # WatermarkDetector — type-level Z-test with homoglyph hardening
├── synonyms.py        # WordNet synonym lookup with LRU cache and similarity filtering
├── bert_synonyms.py   # BERT fill-mask fallback with SBERT similarity gate
├── main.py            # CLI entry point (watermark / detect commands)
├── setup_nltk.py      # Download NLTK corpora (run once)
├── setup_models.py    # Pre-download BERT + SBERT models (run once)
├── test_local.py      # Smoke test — no data required
├── Dockerfile         # GPU container (with BERT-LS)
├── Dockerfile.cpu     # CPU-only container (WordNet only)
└── requirements.txt   # Python dependencies
```

## Setup

### Requirements

- Python 3.8+
- PyTorch with CUDA (for BERT-LS; CPU-only mode available without it)
- See `requirements.txt` for all dependencies

```bash
pip install -r requirements.txt
```

Download NLTK data (one-time):

```bash
python setup_nltk.py
```

Pre-download BERT and SBERT models (one-time, optional — required for BERT-LS fallback):

```bash
python setup_models.py
```

### Data

This repository does **not** include PAN competition data. Download the official data from the [PAN 2026 task page](https://pan.webis.de/clef26/pan26-web/text-watermarking.html) and place `.jsonl` files in an input directory.

Each record must have the schema:
```json
{"id": "...", "text": "..."}
```

## Usage

### Watermark texts

```bash
python main.py watermark <input_dir> <output_dir>
```

Output: `<output_dir>/watermarked-text.jsonl` with `{"id": "...", "text": "<watermarked>"}`.

### Detect watermarks

```bash
python main.py detect <input_dir> <output_dir>
```

Output: `<output_dir>/detected-text.jsonl` with `{"id": "...", "label": 1.0 or 0.0}`.

### Quick smoke test

```bash
python test_local.py
```

Watermarks a sample EU Parliament speech sentence, detects it, verifies shuffle robustness, and prints the green word breakdown.

## Docker (TIRA)

The Dockerfiles target the [TIRA](https://www.tira.io/) evaluation platform.

**GPU (with BERT-LS):**
```bash
docker build -t pan26-watermark .
docker run --gpus all pan26-watermark python /main.py watermark /input /output
```

**CPU-only (WordNet only, no BERT-LS):**
```bash
docker build -f Dockerfile.cpu -t pan26-watermark-cpu .
docker run pan26-watermark-cpu python /main.py watermark /input /output
```

## Key Hyperparameters

All parameters are in [`config.py`](config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WATERMARK_KEY_1` | env var `WATERMARK_KEY_1` | HMAC key for Key-1 |
| `WATERMARK_KEY_2` | env var `WATERMARK_KEY_2` | HMAC key for Key-2 |
| `Z_THRESHOLD` | 1.5 | Detection threshold — both keys must exceed this |
| `Z_WATERMARK_TARGET` | 2.5 | Embedding target Z-score (safety margin above threshold) |
| `P0_GREEN` | 0.513 | Empirical null green fraction for Key-1 |
| `P0_GREEN_2` | 0.508 | Empirical null green fraction for Key-2 |
| `USE_BERT_LS` | `True` | Enable BERT-LS fallback (set `False` for WordNet-only) |
| `MAX_TYPES_TO_SUBSTITUTE` | 12 | Max word types substituted per embedding stage |

**Using your own keys:** set environment variables before running:

```bash
export WATERMARK_KEY_1="your-secret-key-1"
export WATERMARK_KEY_2="your-secret-key-2"
```

Watermarks embedded with one key set are not detectable by a detector using different keys.

## Citation

```bibtex
@inproceedings{ibrahim:2026,
  author    = {Momen Ibrahim},
  title     = {Post-Hoc Synonym Watermarking with Dual-Key Type-Level Detection:
               A Precision--Robustness Trade-off at PAN 2026},
  booktitle = {CLEF 2026 Working Notes},
  year      = {2026},
  publisher = {CEUR-WS.org},
}
```

This system adapts the green-list watermarking framework of [Kirchenbauer et al. (2023)](https://proceedings.mlr.press/v202/kirchenbauer23a.html) to a post-hoc synonym substitution setting, related to [PostMark (Chang et al., 2024)](https://doi.org/10.18653/v1/2024.emnlp-main.506).

## License

Code: MIT License. PAN competition data is subject to the [PAN data terms](https://pan.webis.de/).
