"""
build_pairwise_dataset.py — Step 3: Build pairwise reranker training data.

Reads the candidate JSONL produced in Step 1, scores every candidate with
GLEU, then generates all ordered (A, B) pairs with a binary label indicating
which candidate is better. Pairs with very similar scores are discarded to
reduce noise in the training signal.

Usage:
    python build_pairwise_dataset.py \
        --input_file  data/candidates.jsonl \
        --output_file data/pairwise_train.jsonl
"""

import argparse
import itertools
from utils import load_jsonl, save_jsonl, compute_gleu


# ──────────────────────────────────────────────
# WHY PAIRWISE RANKING?
# ──────────────────────────────────────────────
# A pointwise approach (train a model to output an absolute quality score)
# is hard because absolute quality is ill-defined and data is sparse.
#
# Pairwise learning-to-rank instead asks a simpler question:
#   "Given source S, is correction A better than correction B?"
# This binary classification is much easier to learn and generalises well.
# The idea comes from RankSVM / LambdaRank and has been applied successfully
# to machine translation (Callison-Burch et al., 2010) and GEC reranking.

# ──────────────────────────────────────────────
# WHY SKIP PAIRS WITH SMALL SCORE DIFFERENCES?
# ──────────────────────────────────────────────
# When GLEU(A) ≈ GLEU(B), the difference may be within noise: slight wording
# variation that is equally acceptable, or floating-point imprecision.
# Training on such near-tie pairs can confuse the reranker by assigning a
# label to what is essentially a coin-flip.  We filter them out with a
# minimum margin threshold (default 0.05).

MIN_MARGIN = 0.05  # skip pair if |score_A − score_B| < this threshold


def build_pairs_for_sentence(record: dict) -> list[dict]:
    """
    Given one record from candidates.jsonl, return a list of pairwise examples.

    The gold reference is injected as an additional candidate with logprob=0.0
    (we only use logprob at inference, not here).  Including the reference
    ensures the reranker learns that the reference is always the best option
    when it appears in the candidate pool (useful for calibration).
    """
    source    = record["source"]
    reference = record["reference"]
    raw_cands = record["candidates"]   # [{"text": ..., "logprob": ...}, ...]

    # Add reference as an extra candidate ONLY if not already present.
    # If mT5 generated the exact reference string, adding it again would create
    # a (reference, reference) pair with an undefined label and distort scoring.
    existing_texts = {c["text"].strip() for c in raw_cands}
    if reference.strip() not in existing_texts:
        all_candidates = [{"text": reference, "logprob": 0.0}] + raw_cands
    else:
        all_candidates = list(raw_cands)

    # Score every candidate with GLEU against the reference
    scored = []
    for cand in all_candidates:
        gleu = compute_gleu(cand["text"], reference)
        scored.append({"text": cand["text"], "gleu": gleu})

    # Generate all ordered pairs (A, B) — both (A,B) and (B,A) are included
    # so the model sees the label from both directions (data augmentation).
    pairs = []
    for cand_a, cand_b in itertools.permutations(scored, 2):
        score_a = cand_a["gleu"]
        score_b = cand_b["gleu"]

        # Skip near-tie pairs (noisy labels)
        if abs(score_a - score_b) < MIN_MARGIN:
            continue

        label = 1 if score_a > score_b else 0

        pairs.append({
            "source": source,
            "cand_A": cand_a["text"],
            "cand_B": cand_b["text"],
            "label":  label,
        })

    return pairs


def main(args):
    records = load_jsonl(args.input_file)
    print(f"[pairwise] Loaded {len(records)} candidate records from {args.input_file}")

    all_pairs = []
    for rec in records:
        pairs = build_pairs_for_sentence(rec)
        all_pairs.extend(pairs)

    print(f"[pairwise] Generated {len(all_pairs)} pairwise training examples.")
    save_jsonl(all_pairs, args.output_file)
    print(f"[pairwise] Done. Output → {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build pairwise reranker training data.")
    parser.add_argument("--input_file",  default="data/candidates.jsonl",
                        help="Candidate JSONL from generate_candidates.py")
    parser.add_argument("--output_file", default="data/pairwise_train.jsonl",
                        help="Output pairwise JSONL for reranker training.")
    args = parser.parse_args()
    main(args)