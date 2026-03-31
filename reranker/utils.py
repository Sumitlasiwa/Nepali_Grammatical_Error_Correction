"""
utils.py — Shared utilities for the Nepali GEC Reranking Pipeline.

Contains:
  - GLEU sentence-level scoring (used to rank correction candidates)
  - Helper functions for loading/saving JSONL data
"""

import csv
import json
import math
import os
from collections import Counter
from typing import List, Tuple


# ──────────────────────────────────────────────
# GLEU SCORING
# ──────────────────────────────────────────────

def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """
    Build a Counter of all n-grams from a token list.
    E.g. tokens=['a','b','c'], n=2 → Counter({('a','b'):1, ('b','c'):1})
    """
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def compute_gleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """
    Compute sentence-level GLEU (Google-BLEU) between a candidate correction
    and the gold reference sentence.

    What GLEU measures:
    ------------------
    GLEU is a sentence-level metric that measures n-gram overlap between the
    candidate and the reference. Unlike BLEU, it also penalises candidates
    that introduce n-grams NOT present in the reference, making it sensitive
    to both under-generation and over-generation errors.

    Why we use it for ranking GEC candidates:
    -----------------------------------------
    For Grammatical Error Correction we want to reward candidates that are
    close to the gold correction. GLEU gives a scalar in [0, 1] that
    correlates well with human judgements for GEC tasks and is fast to
    compute without any external libraries.

    Tokenisation:
    -------------
    We split on whitespace. This is intentional: Nepali words are naturally
    space-separated, and character-level tokenisation would underweight
    whole-word substitutions (the most common GEC correction type).

    Args:
        candidate : The generated correction string.
        reference : The gold reference string.
        max_n     : Maximum n-gram order (default 4, same as BLEU-4).

    Returns:
        A float in [0.0, 1.0].  1.0 means perfect match.
    """
    cand_tokens = candidate.strip().split()
    ref_tokens  = reference.strip().split()

    if len(cand_tokens) == 0:
        return 0.0

    total_match = 0
    total_cand  = 0
    total_ref   = 0

    for n in range(1, max_n + 1):
        cand_ngrams = _get_ngrams(cand_tokens, n)
        ref_ngrams  = _get_ngrams(ref_tokens,  n)

        # Clipped count: for each n-gram, take min(cand_count, ref_count)
        match = sum(min(cnt, ref_ngrams[gram]) for gram, cnt in cand_ngrams.items())

        total_match += match
        total_cand  += max(len(cand_tokens) - n + 1, 0)
        total_ref   += max(len(ref_tokens)  - n + 1, 0)

    if total_cand == 0 or total_ref == 0:
        return 0.0

    # GLEU = clipped_matches / max(total_cand, total_ref)
    # Using the max denominator penalises both too-short and too-long candidates.
    gleu = total_match / max(total_cand, total_ref)
    return float(gleu)


# ──────────────────────────────────────────────
# JSONL I/O HELPERS
# ──────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    """Read a .jsonl file and return a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_csv(path: str) -> List[dict]:
    """
    Read a CSV file with columns  incorrect_sentence, correct_sentence
    and return a list of {"source": ..., "target": ...} dicts.

    The CSV must have a header row. Column order does not matter as long as
    the names are exactly  incorrect_sentence  and  correct_sentence.
    Blank rows are skipped automatically.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Validate expected columns exist
        required = {"incorrect_sentence", "correct_sentence"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must contain columns {required}. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            src = row["incorrect_sentence"].strip()
            tgt = row["correct_sentence"].strip()
            if src and tgt:          # skip empty rows
                records.append({"source": src, "target": tgt})
    print(f"[utils] Loaded {len(records)} rows from CSV → {path}")
    return records


def load_input(path: str) -> List[dict]:
    """
    Auto-detect file format by extension and load accordingly.
      .csv   → load_csv   (columns: incorrect_sentence, correct_sentence)
      .jsonl → load_jsonl (fields:  source, target)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return load_csv(path)
    elif ext == ".jsonl":
        return load_jsonl(path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Use .csv or .jsonl")


def save_jsonl(records: List[dict], path: str) -> None:
    """Write a list of dicts to a .jsonl file, one JSON object per line."""

    directory = os.path.dirname(path)

    if directory:  # only create if path includes a folder
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[utils] Saved {len(records)} records → {path}")