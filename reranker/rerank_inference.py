"""
rerank_inference.py — Step 5: Full GEC inference pipeline.

Pipeline:
    1. Load fine-tuned mT5 → generate 10 candidate corrections (beam search).
    2. De-duplicate candidates.
    3. Run all candidate pairs through the MuRIL reranker.
    4. Tournament ranking: each candidate earns +1 for every pair it wins.
    5. (Optional) Hybrid score: 0.7 × reranker_win_rate + 0.3 × norm_logprob.
    6. Return the best correction.

Usage (interactive):
    python rerank_inference.py \
        --mt5_model    ./mt5-gec-finetuned \
        --reranker_dir ./muril-reranker

Usage (batch):
    python rerank_inference.py \
        --mt5_model    ./mt5-gec-finetuned \
        --reranker_dir ./muril-reranker \
        --input_file   data/test.jsonl \
        --output_file  data/predictions.jsonl
"""

import argparse
import itertools
import math
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)
from utils import load_jsonl, save_jsonl


# ──────────────────────────────────────────────
# BEAM SEARCH SETTINGS (must match generate_candidates.py)
# ──────────────────────────────────────────────
NUM_BEAMS       = 10
NUM_RETURN_SEQS = 10
MAX_GEN_LEN     = 128

# ──────────────────────────────────────────────
# HYBRID SCORING WEIGHTS
# ──────────────────────────────────────────────
# Why combine generator log-prob and reranker score?
# --------------------------------------------------
# The generator (mT5) is trained to maximise the likelihood of the corrected
# sentence, so its log-probability is a good measure of *fluency*.
# The reranker (MuRIL) is trained to prefer the more *accurate* correction.
# These two signals are complementary:
#   • High log-prob but low reranker score  → fluent but inaccurate
#   • Low log-prob but high reranker score  → accurate but slightly disfluent
# Mixing them (0.7 reranker + 0.3 fluency) consistently outperforms either
# alone, as shown in MT reranking literature (Salimans et al., 2022;
# Fernandes et al., 2022).
RERANKER_WEIGHT  = 0.7
LOGPROB_WEIGHT   = 0.3


# ══════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════

def load_generator(model_path: str, device: torch.device):
    print(f"[inference] Loading mT5 generator from {model_path} …")
    tok   = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).eval().to(device)
    return tok, model


def load_reranker(reranker_dir: str, device: torch.device):
    print(f"[inference] Loading MuRIL reranker from {reranker_dir} …")
    tok   = AutoTokenizer.from_pretrained(reranker_dir)
    model = AutoModelForSequenceClassification.from_pretrained(reranker_dir).eval().to(device)
    return tok, model


# ══════════════════════════════════════════════
# CANDIDATE GENERATION
# ══════════════════════════════════════════════

def generate_candidates(
    source: str,
    gen_tok,
    gen_model,
    device: torch.device,
) -> list[dict]:
    """
    Generate up to NUM_RETURN_SEQS unique candidates for `source`.
    Returns list of {"text": str, "logprob": float}.
    """
    inputs = gen_tok(
        source,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_GEN_LEN,
    ).to(device)

    with torch.no_grad():
        out = gen_model.generate(
            **inputs,
            num_beams=NUM_BEAMS,
            num_return_sequences=NUM_RETURN_SEQS,
            max_length=MAX_GEN_LEN,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    texts  = gen_tok.batch_decode(out.sequences, skip_special_tokens=True)
    scores = out.sequences_scores.cpu().tolist()

    # De-duplicate: keep best log-prob per unique text
    seen: dict[str, float] = {}
    for text, lp in zip(texts, scores):
        text = text.strip()
        if text not in seen or lp > seen[text]:
            seen[text] = lp

    return [{"text": t, "logprob": lp} for t, lp in seen.items()]


# ══════════════════════════════════════════════
# PAIRWISE RERANKING
# ══════════════════════════════════════════════

def rerank_candidates(
    source: str,
    candidates: list[dict],
    rr_tok,
    rr_model,
    device: torch.device,
    use_hybrid: bool = True,
) -> str:
    """
    Rank `candidates` using tournament pairwise scoring.

    Tournament ranking:
    -------------------
    Each candidate starts with 0 wins.  For every ordered pair (A, B) we ask
    the reranker "is A better than B?"  If the reranker predicts label=1
    (A wins), candidate A gets +1.  The candidate with the most wins is chosen.
    This is equivalent to a round-robin tournament and is more robust than
    single-elimination because every candidate is compared against every other.

    Hybrid scoring (optional):
    --------------------------
    After computing win counts we normalise them to [0,1] and blend with the
    normalised generator log-probability.  See HYBRID SCORING WEIGHTS above.
    """
    n = len(candidates)
    if n == 1:
        return candidates[0]["text"]

    # ── Win counter ────────────────────────────────────────────────────────
    wins = [0] * n

    # Build all (i, j) ordered pairs in a single batch for speed
    pair_indices = list(itertools.permutations(range(n), 2))

    batch_sources = []
    batch_second  = []
    for i, j in pair_indices:
        batch_sources.append(source)
        batch_second.append(
            f"{candidates[i]['text']} [SEP] {candidates[j]['text']}"
        )

    # Tokenise the whole batch
    enc = rr_tok(
        batch_sources,
        batch_second,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = rr_model(**enc).logits          # (num_pairs, 2)
        probs  = F.softmax(logits, dim=-1)        # (num_pairs, 2)
        # prob[:, 1] = P(A is better than B)

    for k, (i, j) in enumerate(pair_indices):
        if probs[k, 1].item() > 0.5:
            wins[i] += 1

    # ── Hybrid score ────────────────────────────────────────────────────────
    if use_hybrid:
        win_scores = [w / max(n - 1, 1) for w in wins]  # normalise to [0,1]

        # Normalise log-probs: shift so max=0, then map to [0,1] via softmax
        logprobs = torch.tensor([c["logprob"] for c in candidates], dtype=torch.float)
        norm_lp  = F.softmax(logprobs, dim=0).tolist()

        final_scores = [
            RERANKER_WEIGHT * ws + LOGPROB_WEIGHT * lp
            for ws, lp in zip(win_scores, norm_lp)
        ]
        best_idx = final_scores.index(max(final_scores))
    else:
        best_idx = wins.index(max(wins))

    return candidates[best_idx]["text"]


# ══════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════

def correct_sentence(
    source: str,
    gen_tok, gen_model,
    rr_tok,  rr_model,
    device: torch.device,
    use_hybrid: bool = True,
) -> str:
    """End-to-end correction for a single source sentence."""
    candidates = generate_candidates(source, gen_tok, gen_model, device)
    best       = rerank_candidates(source, candidates, rr_tok, rr_model, device, use_hybrid)
    return best


# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Using device: {device}")

    gen_tok, gen_model = load_generator(args.mt5_model,    device)
    rr_tok,  rr_model  = load_reranker (args.reranker_dir, device)

    if args.input_file:
        # ── Batch mode ──────────────────────────────────────────────────────
        records = load_jsonl(args.input_file)
        output  = []
        for i, rec in enumerate(records):
            source    = rec["source"]
            reference = rec.get("target", "")
            prediction = correct_sentence(
                source, gen_tok, gen_model, rr_tok, rr_model, device,
                use_hybrid=not args.no_hybrid,
            )
            output.append({
                "source":     source,
                "reference":  reference,
                "prediction": prediction,
            })
            if (i + 1) % 50 == 0:
                print(f"[inference] {i+1}/{len(records)} done …")

        save_jsonl(output, args.output_file)
        print(f"[inference] Predictions saved → {args.output_file}")

    else:
        # ── Interactive mode ─────────────────────────────────────────────────
        print("\n[inference] Interactive mode. Type a Nepali sentence (or 'quit' to exit).")
        while True:
            src = input("\nSource: ").strip()
            if src.lower() in ("quit", "exit", "q"):
                break
            if not src:
                continue
            best = correct_sentence(
                src, gen_tok, gen_model, rr_tok, rr_model, device,
                use_hybrid=not args.no_hybrid,
            )
            print(f"Correction: {best}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nepali GEC inference with reranker.")
    parser.add_argument("--mt5_model",    default="./mt5-gec-finetuned",
                        help="Path to fine-tuned mT5 model.")
    parser.add_argument("--reranker_dir", default="./muril-reranker",
                        help="Path to trained MuRIL reranker.")
    parser.add_argument("--input_file",   default=None,
                        help="(Optional) Input JSONL for batch inference.")
    parser.add_argument("--output_file",  default="data/predictions.jsonl",
                        help="Output JSONL for batch predictions.")
    parser.add_argument("--no_hybrid",    action="store_true",
                        help="Disable hybrid scoring (use tournament wins only).")
    args = parser.parse_args()
    main(args)
