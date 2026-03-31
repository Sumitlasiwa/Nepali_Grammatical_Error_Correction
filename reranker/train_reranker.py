"""
train_reranker.py — Step 4: Train a MuRIL cross-encoder pairwise reranker.

Architecture:
    google/muril-base-cased  (Multilingual Representations for Indian Languages)
    → AutoModelForSequenceClassification (2-class: A better / B better)

Input format fed to the tokeniser:
    [CLS] source [SEP] candidate_A [SEP] candidate_B [SEP]

Labels:
    1 → candidate A is the better correction
    0 → candidate B is the better correction

Usage:
    python train_reranker.py \
        --train_file      data/pairwise_train.jsonl \
        --output_dir      ./muril-reranker \
        --wandb_project   nepali-gec-reranker \
        --wandb_run_name  muril-run1

Before running, log in to wandb once:
    pip install wandb
    wandb login
"""

import argparse
import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from utils import load_jsonl


# ──────────────────────────────────────────────
# CROSS-ENCODER ARCHITECTURE
# ──────────────────────────────────────────────
# A "cross-encoder" means ALL three inputs (source, cand_A, cand_B) are
# concatenated and fed into a single BERT-style encoder at once.  This lets
# every token attend to every other token via full self-attention, allowing
# the model to perform fine-grained comparison.
#
# WHY MANUAL TOKEN-ID CONSTRUCTION (not f-string "[SEP]"):
# ─────────────────────────────────────────────────────────
# Embedding "[SEP]" as a plain string inside an f-string and passing it to
# the tokeniser causes it to be subword-tokenised (e.g. ['[', '##SE', '##P',
# ']']).  The model never sees a real separator token (ID=102), so all three
# segments blur into one garbled sequence and the model cannot tell cand_A
# from cand_B.  The fix is to call tokenizer.encode() on each piece
# separately with add_special_tokens=False, then manually insert the real
# CLS/SEP token IDs obtained from tokenizer.cls_token_id / .sep_token_id.
#
# WHY MuRIL?
# MuRIL (Khanuja et al., 2021) was pre-trained on 17 Indian languages plus
# transliterated text. Nepali shares significant vocabulary with Hindi and
# Sanskrit, so MuRIL's representations are far better suited than plain
# multilingual-BERT for Nepali GEC.

MODEL_NAME  = "google/muril-large-cased"
MAX_LENGTH  = 256
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 2e-5


def tokenise_batch(batch, tokenizer):
    """
    Tokenise a batch of (source, cand_A, cand_B) triples into a cross-encoder
    input with the layout:

        [CLS] source [SEP] cand_A [SEP] cand_B [SEP]

    Each piece is encoded independently with add_special_tokens=False, then
    the real CLS/SEP token IDs are inserted manually.  This is critical:
    embedding the string "[SEP]" inside an f-string causes the tokeniser to
    split it into subword pieces (['[', '##SE', '##P', ']']), so the model
    never sees a genuine separator boundary.

    Segment IDs (token_type_ids):
        0 → source tokens  (first segment)
        1 → cand_A + cand_B tokens  (second segment)

    Truncation strategy:
        We truncate the whole sequence to MAX_LENGTH after concatenation.
        For very long inputs this may clip the tail of cand_B; a smarter
        per-segment truncation can be added later if needed.
    """
    all_input_ids      = []
    all_attention_masks = []
    all_token_type_ids  = []

    CLS = tokenizer.cls_token_id
    SEP = tokenizer.sep_token_id

    for src, a, b in zip(batch["source"], batch["cand_A"], batch["cand_B"]):
        # Encode each segment without special tokens so we control placement
        src_ids = tokenizer.encode(src, add_special_tokens=False)
        a_ids   = tokenizer.encode(a,   add_special_tokens=False)
        b_ids   = tokenizer.encode(b,   add_special_tokens=False)

        # Build: [CLS] source [SEP] cand_A [SEP] cand_B [SEP]
        input_ids = (
            [CLS]
            + src_ids + [SEP]
            + a_ids   + [SEP]
            + b_ids   + [SEP]
        )

        # Truncate to MAX_LENGTH (keeps the head; tail of cand_B may be cut)
        input_ids = input_ids[:MAX_LENGTH]

        # Segment 0: CLS + source + SEP  |  Segment 1: the rest
        n_seg0 = len(src_ids) + 2           # +2 for CLS and first SEP
        token_type_ids = (
            [0] * min(n_seg0, len(input_ids))
            + [1] * max(0, len(input_ids) - n_seg0)
        )[:len(input_ids)]                  # ensure same length as input_ids

        attention_mask = [1] * len(input_ids)

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_token_type_ids.append(token_type_ids)

    return {
        "input_ids":       all_input_ids,
        "attention_mask":  all_attention_masks,
        "token_type_ids":  all_token_type_ids,
        "labels":          batch["label"],
    }


def compute_metrics(eval_pred):
    """
    Compute accuracy for validation.

    Metrics logged:
      eval/accuracy      — overall fraction of correctly ranked pairs
      eval/accuracy_A    — when A is truly better, how often do we predict A?
      eval/accuracy_B    — when B is truly better, how often do we predict B?
    """
    import numpy as np
    logits, labels = eval_pred
    preds  = logits.argmax(axis=-1)
    labels = labels.astype(int)

    overall_acc = float((preds == labels).mean())

    mask_a = labels == 1
    mask_b = labels == 0
    acc_a = float((preds[mask_a] == labels[mask_a]).mean()) if mask_a.any() else 0.0
    acc_b = float((preds[mask_b] == labels[mask_b]).mean()) if mask_b.any() else 0.0

    return {
        "accuracy":   overall_acc,
        "accuracy_A": acc_a,
        "accuracy_B": acc_b,
    }


class WandbAccuracyCallback:
    """
    Custom HuggingFace TrainerCallback that re-logs eval metrics at every
    evaluation step so train/loss and eval/accuracy share the same x-axis
    in wandb charts.
    """
    from transformers import TrainerCallback

    class _Callback(TrainerCallback):  # type: ignore
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None:
                return
            epoch = int(state.epoch) if state.epoch is not None else state.global_step
            wandb.log({
                "epoch":            epoch,
                "eval/accuracy":    metrics.get("eval_accuracy",   0.0),
                "eval/accuracy_A":  metrics.get("eval_accuracy_A", 0.0),
                "eval/accuracy_B":  metrics.get("eval_accuracy_B", 0.0),
                "eval/loss":        metrics.get("eval_loss",        0.0),
            }, step=state.global_step)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            train_logs = {
                f"train/{k}": v for k, v in logs.items()
                if k in ("loss", "learning_rate", "grad_norm")
            }
            if train_logs:
                wandb.log(train_logs, step=state.global_step)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Using device: {device}")

    wandb.init(
        project = args.wandb_project,
        name    = args.wandb_run_name,
        config  = {
            "model":       MODEL_NAME,
            "epochs":      EPOCHS,
            "batch_size":  BATCH_SIZE,
            "lr":          LR,
            "max_length":  MAX_LENGTH,
            "train_file":  args.train_file,
        },
        save_code=True,
    )

    # ── Load data ──────────────────────────────────────────────────────────
    raw = load_jsonl(args.train_file)
    print(f"[train] Loaded {len(raw)} pairwise examples.")

    dataset    = Dataset.from_list(raw).train_test_split(test_size=0.1, seed=10)
    train_data = dataset["train"]
    val_data   = dataset["test"]

    wandb.config.update({"train_size": len(train_data), "val_size": len(val_data)})

    # ── Tokeniser & model ──────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    # ── Sanity-check: print a decoded example before training ──────────────
    sample = train_data[0]
    CLS = tokenizer.cls_token_id
    SEP = tokenizer.sep_token_id
    src_ids = tokenizer.encode(sample["source"], add_special_tokens=False)
    a_ids   = tokenizer.encode(sample["cand_A"], add_special_tokens=False)
    b_ids   = tokenizer.encode(sample["cand_B"], add_special_tokens=False)
    check_ids = ([CLS] + src_ids + [SEP] + a_ids + [SEP] + b_ids + [SEP])[:MAX_LENGTH]
    print("\n[train] === Tokenisation sanity check ===")
    print(tokenizer.decode(check_ids))
    print(f"[train] label: {sample['label']}")
    print("[train] =====================================\n")

    # ── Tokenise datasets ──────────────────────────────────────────────────
    tokenise_fn = lambda batch: tokenise_batch(batch, tokenizer)

    train_dataset = train_data.map(
        tokenise_fn, batched=True, num_proc=4, batch_size=1000,
        remove_columns=["source", "cand_A", "cand_B", "label"],
    )
    val_dataset = val_data.map(
        tokenise_fn, batched=True, num_proc=4, batch_size=1000,
        remove_columns=["source", "cand_A", "cand_B", "label"],
    )

    # ── Step / eval scheduling ─────────────────────────────────────────────
    EVALS_PER_EPOCH = 4
    steps_per_epoch = max(1, len(train_dataset) // BATCH_SIZE)
    eval_steps      = max(1, steps_per_epoch // EVALS_PER_EPOCH)
    logging_steps   = max(1, eval_steps // 2)

    print(f"[train] steps_per_epoch={steps_per_epoch} | "
          f"eval_steps={eval_steps} | logging_steps={logging_steps}")

    # ── Training arguments ─────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = args.output_dir,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate               = LR,
        warmup_ratio                = 0.1,
        weight_decay                = 0.01,
        eval_strategy               = "steps",
        eval_steps                  = eval_steps,
        save_strategy               = "steps",
        save_steps                  = eval_steps,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "accuracy",
        greater_is_better           = True,
        fp16                        = torch.cuda.is_available(),
        report_to                   = "wandb",
        logging_strategy            = "steps",
        logging_steps               = logging_steps,
        run_name                    = args.wandb_run_name,
        seed                        = 42,
    )

    data_collator     = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy_callback = WandbAccuracyCallback._Callback()

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        tokenizer       = tokenizer,
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
        callbacks       = [accuracy_callback],
    )

    print("[train] Starting training …")
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[train] Best model saved → {args.output_dir}")

    wandb.finish()
    print("[train] wandb run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MuRIL pairwise reranker.")
    parser.add_argument("--train_file",     default="data/pairwise_train.jsonl")
    parser.add_argument("--output_dir",     default="./muril-reranker")
    parser.add_argument("--wandb_project",  default="nepali-gec-reranker")
    parser.add_argument("--wandb_run_name", default="muril-reranker-run1")
    args = parser.parse_args()
    main(args)