"""
generate_candidates.py — Step 1: Generate correction candidates with fine-tuned mT5.

For each source (incorrect) sentence we run beam search to produce 5 diverse
candidate corrections. The candidates, along with their generator log-probabilities,
are written to a JSONL file for downstream reranking.

Input file formats supported:
  CSV  — columns: incorrect_sentence, correct_sentence  (header row required)
  JSONL — fields:  source, target
  Format is auto-detected from the file extension.

Usage (CSV):
    python generate_candidates.py \
        --model_path  ./mt5-gec-finetuned \
        --input_file  data/test.csv \
        --output_file data/candidates.jsonl

Usage (JSONL):
    python generate_candidates.py \
        --model_path  ./mt5-gec-finetuned \
        --input_file  data/test.jsonl \
        --output_file data/candidates.jsonl
"""

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import load_input, save_jsonl


# ──────────────────────────────────────────────
# BEAM SEARCH — HOW IT WORKS
# ──────────────────────────────────────────────
# Standard greedy decoding always keeps only the single most-probable next token,
# so it can easily get trapped in a local optimum.
#
# Beam search keeps a "beam" of the top-k hypotheses at every decoding step and
# expands all of them simultaneously. With num_beams=5 and
# num_return_sequences=5, the decoder returns the 5 highest-scoring complete
# sequences — a leaner pool that's faster to generate and rerank while still
# covering the main correction hypotheses.
#
# Why 5 beams?  Fewer beams → faster inference, lower VRAM usage. For shorter
# sentences (max_length=64) the correction space is smaller, so 5 beams gives
# good coverage without the overhead of 10.

NUM_BEAMS           = 10
NUM_RETURN_SEQS     = 10
MAX_LENGTH          = 64
BATCH_SIZE          = 8          # adjust to fit P100 VRAM


def load_model(model_path: str, device: torch.device):
    """Load the fine-tuned mT5 tokeniser and model onto `device`."""
    print(f"[generate] Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    model.to(device)
    return tokenizer, model


def generate_for_batch(
    sources: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> list[list[dict]]:
    """
    Run beam search on a batch of source sentences.

    Returns a list (one entry per source sentence) of lists of dicts:
        [{"text": str, "logprob": float}, ...]

    Why we save generator log-probability:
    ---------------------------------------
    The model assigns a score (sum of log token-probabilities) to each
    completed beam sequence. This score reflects how "fluent" the sequence is
    according to the mT5 language model. We save it so that at inference time
    we can optionally combine it with the reranker's pairwise score:

        final_score = 0.7 * reranker_score + 0.3 * normalised_logprob

    This hybrid strategy typically beats either signal alone because the
    generator captures fluency while the reranker captures correction accuracy.
    """
    inputs = tokenizer(
        sources,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=NUM_BEAMS,
            num_return_sequences=NUM_RETURN_SEQS,
            max_length=MAX_LENGTH,
            early_stopping=True,
            # Return beam scores so we can extract log-probabilities
            output_scores=True,
            return_dict_in_generate=True,
        )

    # outputs.sequences  shape: (batch * NUM_RETURN_SEQS, seq_len)
    # outputs.sequences_scores shape: (batch * NUM_RETURN_SEQS,)  — sum of log-probs
    sequences = outputs.sequences
    scores    = outputs.sequences_scores.cpu().tolist()   # raw log-probs (negative floats)

    decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)

    # Re-group by source sentence
    batch_results = []
    for i, src in enumerate(sources):
        start = i * NUM_RETURN_SEQS
        end   = start + NUM_RETURN_SEQS

        # ── De-duplication ────────────────────────────────────────────────────
        # Beam search sometimes produces identical strings (e.g. when the model
        # is very confident). Duplicate candidates add no value to the reranker
        # training set and waste inference time, so we remove them here while
        # preserving the best (highest) log-probability for each unique text.
        seen: dict[str, float] = {}
        for text, lp in zip(decoded[start:end], scores[start:end]):
            text = text.strip()
            if text not in seen or lp > seen[text]:
                seen[text] = lp

        candidates = [{"text": t, "logprob": round(lp, 6)} for t, lp in seen.items()]
        batch_results.append(candidates)

    return batch_results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate] Using device: {device}")

    tokenizer, model = load_model(args.model_path, device)

    # Input file: CSV (incorrect_sentence,correct_sentence) or JSONL (source,target)
    # load_input() auto-detects the format from the file extension.
    records = load_input(args.input_file)
    print(f"[generate] Loaded {len(records)} sentences from {args.input_file}")

    output_records = []

    # Process in batches for efficiency
    for batch_start in range(0, len(records), BATCH_SIZE):
        batch = records[batch_start : batch_start + BATCH_SIZE]
        sources    = [r["source"] for r in batch]
        references = [r["target"] for r in batch]

        candidates_per_src = generate_for_batch(sources, tokenizer, model, device)

        for src, ref, candidates in zip(sources, references, candidates_per_src):
            output_records.append({
                "source":     src,
                "reference":  ref,
                "candidates": candidates,
            })

        print(f"[generate] Processed {min(batch_start + BATCH_SIZE, len(records))}"
              f" / {len(records)} sentences …")

    save_jsonl(output_records, args.output_file)
    print(f"[generate] Done. Output → {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GEC candidates with mT5.")
    parser.add_argument("--model_path",  default="./mt5-gec-finetuned",
                        help="Path to fine-tuned mT5 model directory.")
    parser.add_argument("--input_file",  default="data/test.csv",
                        help="Input file: CSV with columns incorrect_sentence,correct_sentence "
                             "OR JSONL with fields source,target. Format is auto-detected.")
    parser.add_argument("--output_file", default="data/candidates.jsonl",
                        help="Output JSONL with generated candidates.")
    args = parser.parse_args()
    main(args)
