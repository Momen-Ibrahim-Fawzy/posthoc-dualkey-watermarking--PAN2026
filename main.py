#!/usr/bin/env python3
"""
PAN@CLEF 2026 Text Watermarking — CLI entry point.

Usage:

  Watermark:
    ./main.py watermark <input_directory> <output_directory>

  Detect:
    ./main.py detect <input_directory> <output_directory>

The <input_directory> must contain one or more *.jsonl files with records
of the form {"id": "...", "text": "..."}.  The output is written to a single
*.jsonl file inside <output_directory>:
  - watermark → watermarked-text.jsonl  {"id": ..., "text": <watermarked>}
  - detect    → detected-text.jsonl     {"id": ..., "label": 0.0 or 1.0}
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from watermarker import TextWatermarker
from detector import WatermarkDetector


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_data(directory: Path) -> pd.DataFrame:
    """Load all *.jsonl records from *directory* into a DataFrame."""
    records = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return pd.DataFrame.from_records(records)


def _save_jsonl(df: pd.DataFrame, path: Path) -> None:
    """Write *df* rows to a newline-delimited JSON file at *path*."""
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(str(path), "w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            json.dump(row.to_dict(), fh, ensure_ascii=True)
            fh.write("\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.group()
def cli() -> None:
    """PAN@CLEF 2026 text watermarking system."""


@cli.command()
@click.argument("input_directory", type=Path)
@click.argument("output_directory", type=Path)
def watermark(input_directory: Path, output_directory: Path) -> None:
    """Embed watermarks into every text in INPUT_DIRECTORY.

    Reads *.jsonl files, applies green-list synonym substitution to each
    text, and writes the watermarked texts to OUTPUT_DIRECTORY/watermarked-text.jsonl.
    """
    data = _load_data(input_directory)
    wm = TextWatermarker()

    watermarked: list[str] = []
    for text in tqdm(data["text"], desc="Watermarking", unit="doc"):
        watermarked.append(wm.watermark(text))

    data["text"] = watermarked
    _save_jsonl(data, output_directory / "watermarked-text.jsonl")
    print(f"Watermarked {len(data)} documents → {output_directory / 'watermarked-text.jsonl'}")


@cli.command()
@click.argument("input_directory", type=Path)
@click.argument("output_directory", type=Path)
def detect(input_directory: Path, output_directory: Path) -> None:
    """Detect watermarks in texts from INPUT_DIRECTORY (possibly attacked).

    Reads *.jsonl files, runs the Z-test detector on each text, and writes
    labels (1.0 = watermarked, 0.0 = not watermarked) to
    OUTPUT_DIRECTORY/detected-text.jsonl.
    """
    data = _load_data(input_directory)
    detector = WatermarkDetector()

    labels: list[float] = []
    for text in tqdm(data["text"], desc="Detecting ", unit="doc"):
        labels.append(detector.detect(text))

    data["label"] = labels

    # The evaluator expects id + label (and optional truth_label), not the text.
    if "text" in data.columns:
        del data["text"]

    _save_jsonl(data, output_directory / "detected-text.jsonl")
    print(f"Detected {len(data)} documents → {output_directory / 'detected-text.jsonl'}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
