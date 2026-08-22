MODEL_VERSION = "1.0.0"

import gc
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    MBart50Tokenizer,
)


# =====================================================
# Device
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# Model paths
# =====================================================

MT5_MODEL_PATH = "AIsumit123/mt5-base-nepali-gec-stage2"
MT5_RERANKER_PATH = "AIsumit123/muril-nepali-gec-reranker-mt5"

MBART_MODEL_PATH = "tuyal/Stage2FFTmBart"
MBART_RERANKER_PATH = "AIsumit123/muril-reranker-mbart"

NLLB_MODEL_PATH = "tuyal/Stage2FFTnllb200"
NLLB_RERANKER_PATH = "tuyal/nllbReranker"


# =====================================================
# Prefixes
# =====================================================

mt5_prefix = "Correct sentence: "
mbart_prefix = ""
nllb_prefix = ""


# =====================================================
# Model configuration
# =====================================================

MODELS_CONFIG = {
    "mt5": {
        "model_path": MT5_MODEL_PATH,
        "prefix": mt5_prefix,
        "rerank_path": MT5_RERANKER_PATH,
        "tokenizer_type": "mt5",
        "seq_class": AutoModelForSeq2SeqLM,
        "rerank_class": AutoModelForSequenceClassification,
    },

    "mbart": {
        "model_path": MBART_MODEL_PATH,
        "prefix": mbart_prefix,
        "rerank_path": MBART_RERANKER_PATH,
        "tokenizer_type": "mbart50",
        "seq_class": AutoModelForSeq2SeqLM,
        "rerank_class": AutoModelForSequenceClassification,
    },

    "nllb": {
        "model_path": NLLB_MODEL_PATH,
        "prefix": nllb_prefix,
        "rerank_path": NLLB_RERANKER_PATH,
        "tokenizer_type": "nllb",
        "seq_class": AutoModelForSeq2SeqLM,
        "rerank_class": AutoModelForSequenceClassification,
    },
}


# =====================================================
# Loaded model cache
# =====================================================

LOADED = {}


# =====================================================
# Unload models
# =====================================================

def _unload_all_except(keep_choice=None):
    """
    Unload all cached models except keep_choice.
    This helps reduce VRAM usage.
    """

    to_delete = [
        key for key in LOADED.keys()
        if key != keep_choice
    ]

    for key in to_delete:
        entry = LOADED.pop(key)

        for name, obj in entry.items():
            try:
                del obj
            except Exception:
                pass

    # Python garbage collection
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# =====================================================
# Load tokenizer
# =====================================================

def _load_tokenizer(choice, cfg):
    """
    Load the correct tokenizer.

    We intentionally use slow tokenizers for SentencePiece-based
    models to avoid the byte-fallback conversion warning.
    """

    tokenizer_type = cfg["tokenizer_type"]

    # -----------------------------
    # mBART
    # -----------------------------
    if tokenizer_type == "mbart50":

        tokenizer = MBart50Tokenizer.from_pretrained(
            "facebook/mbart-large-50"
        )

        tokenizer.src_lang = "ne_NP"
        tokenizer.tgt_lang = "ne_NP"

        return tokenizer

    # -----------------------------
    # NLLB
    # -----------------------------
    elif tokenizer_type == "nllb":

        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            use_fast=False
        )

        try:
            tokenizer.src_lang = "npi_Deva"
            tokenizer.tgt_lang = "npi_Deva"
        except Exception:
            pass

        return tokenizer

    # -----------------------------
    # mT5
    # -----------------------------
    elif tokenizer_type == "mt5":

        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_path"],
            use_fast=False
        )

        return tokenizer

    else:
        raise ValueError(
            f"Unknown tokenizer type: {tokenizer_type}"
        )


# =====================================================
# Load model choice
# =====================================================

def _load_choice(choice):
    """
    Load seq2seq model, tokenizer, reranker and reranker
    tokenizer for the selected model.

    Only one model choice is kept in memory at a time.
    """

    # Already loaded
    if choice in LOADED:
        return LOADED[choice]

    # Validate choice
    if choice not in MODELS_CONFIG:
        raise ValueError(
            f"Unknown model choice: {choice}. "
            f"Available choices: {list(MODELS_CONFIG.keys())}"
        )

    # Unload previous model
    _unload_all_except(keep_choice=None)

    cfg = MODELS_CONFIG[choice]

    print(f"Loading {choice} model...")
    print(f"Device: {device}")

    # =================================================
    # Load main tokenizer
    # =================================================

    tokenizer = _load_tokenizer(choice, cfg)

    # =================================================
    # Load main seq2seq model
    # =================================================

    print(f"Loading generator model: {cfg['model_path']}")

    seq_model = cfg["seq_class"].from_pretrained(
        cfg["model_path"]
    ).to(device)

    seq_model.eval()

    # =================================================
    # Load reranker tokenizer
    # =================================================

    print(f"Loading reranker tokenizer: {cfg['rerank_path']}")

    rerank_tokenizer = AutoTokenizer.from_pretrained(
        cfg["rerank_path"],
        trust_remote_code=True,
        use_fast=False
    )

    # =================================================
    # Load reranker
    # =================================================

    print(f"Loading reranker model: {cfg['rerank_path']}")

    rerank_model = cfg["rerank_class"].from_pretrained(
        cfg["rerank_path"],
        trust_remote_code=True
    ).to(device)

    rerank_model.eval()

    # =================================================
    # Store everything
    # =================================================

    entry = {
        "tokenizer": tokenizer,
        "prefix": cfg["prefix"],
        "model": seq_model,
        "rerank_tokenizer": rerank_tokenizer,
        "rerank_model": rerank_model,
    }

    LOADED[choice] = entry

    print(f"{choice} model loaded successfully.")

    return entry


# =====================================================
# Generate candidate corrections
# =====================================================

def generate_candidates(
    input_text,
    model_choice="mt5",
    k=5
):
    """
    Generate multiple possible corrected sentences.
    """

    entry = _load_choice(model_choice)

    prefix = entry["prefix"]
    tokenizer = entry["tokenizer"]
    model = entry["model"]

    # Add model-specific prefix
    prefixed = prefix + input_text

    # Tokenize
    inputs = tokenizer(
        prefixed,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    # Move tensors to GPU/CPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # =================================================
    # Forced BOS token
    # =================================================

    forced_bos_token_id = None

    if model_choice == "mbart":

        try:
            forced_bos_token_id = tokenizer.lang_code_to_id[
                "ne_NP"
            ]
        except Exception:
            forced_bos_token_id = None

    elif model_choice == "nllb":

        try:
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(
                "npi_Deva"
            )
        except Exception:
            forced_bos_token_id = None

    # =================================================
    # Generation
    # =================================================

    with torch.no_grad():

        generation_kwargs = {
            "num_beams": k,
            "num_return_sequences": k,
            "max_new_tokens": inputs["input_ids"].shape[1] + 8,
            "length_penalty": 0.7,
            "repetition_penalty": 1.05,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }

        # Only provide forced_bos_token_id when valid
        if forced_bos_token_id is not None:
            generation_kwargs["forced_bos_token_id"] = (
                forced_bos_token_id
            )

        outputs = model.generate(
            **inputs,
            **generation_kwargs
        )

    # =================================================
    # Decode
    # =================================================

    candidates = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    # Remove whitespace
    candidates = [
        candidate.strip()
        for candidate in candidates
    ]

    # Remove duplicates while preserving order
    candidates = list(dict.fromkeys(candidates))

    return candidates


# =====================================================
# Pairwise comparison
# =====================================================

def compare_pair(
    source,
    cand_A,
    cand_B,
    model_choice="mt5"
):
    """
    Compare two candidate corrections.

    Returns probability that candidate A is better.
    """

    entry = _load_choice(model_choice)

    rerank_tokenizer = entry["rerank_tokenizer"]
    rerank_model = entry["rerank_model"]

    # Tokenize source + candidate pair
    inputs = rerank_tokenizer(
        source,
        f"{cand_A} {rerank_tokenizer.sep_token} {cand_B}",
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    # Move tensors to device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # =================================================
    # Reranker prediction
    # =================================================

    with torch.no_grad():

        logits = rerank_model(
            **inputs
        ).logits

        probs = torch.softmax(
            logits,
            dim=-1
        )

    # Label 1 = candidate A is better
    prob_A = probs[0][1].item()

    return prob_A


# =====================================================
# Tournament reranking
# =====================================================

def rerank_candidates(
    source,
    candidates,
    model_choice="mt5"
):
    """
    Rank candidates using pairwise tournament scoring.
    """

    n = len(candidates)

    # Nothing to rank
    if n == 0:
        return []

    # Only one candidate
    if n == 1:
        return [(candidates[0], 0)]

    scores = [0] * n

    # Compare every pair
    for i in range(n):

        for j in range(i + 1, n):

            prob_A = compare_pair(
                source,
                candidates[i],
                candidates[j],
                model_choice
            )

            if prob_A > 0.5:
                scores[i] += 1
            else:
                scores[j] += 1

    # Sort by number of wins
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


# =====================================================
# Predict best correction
# =====================================================

def predict_output(
    input_text,
    model_choice="mt5"
):
    """
    Return the best grammatical correction.
    """

    input_text = input_text.strip()

    if not input_text:
        return ""

    # Generate candidates
    candidates = generate_candidates(
        input_text,
        model_choice=model_choice,
        k=5
    )

    # Safety fallback
    if not candidates:
        return input_text

    # Rerank candidates
    ranked = rerank_candidates(
        input_text,
        candidates,
        model_choice
    )

    # Return best candidate
    best_sentence = ranked[0][0]

    return best_sentence


# =====================================================
# Predict multiple sentences
# =====================================================

def predict_multiple(
    sentences,
    model_choice="mt5"
):
    """
    Correct multiple sentences.
    """

    outputs = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            outputs.append("")
            continue

        corrected = predict_output(
            sentence,
            model_choice
        )

        outputs.append(corrected)

    return outputs


# =====================================================
# Debug: candidates + scores
# =====================================================

def get_all_candidates_with_scores(
    input_text,
    model_choice="mt5"
):
    """
    Return all generated candidates and their
    tournament scores.
    """

    candidates = generate_candidates(
        input_text,
        model_choice=model_choice,
        k=5
    )

    ranked = rerank_candidates(
        input_text,
        candidates,
        model_choice
    )

    results = []

    for rank, (sentence, score) in enumerate(
        ranked,
        start=1
    ):

        results.append({
            "rank": rank,
            "sentence": sentence,
            "wins": score
        })

    return results
