
MODEL_VERSION = '1.0.0'


import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    MBart50TokenizerFast
)

device = "cuda" if torch.cuda.is_available() else "cpu"
# Model paths and prefixes (kept as config only; models are loaded lazily)
MT5_MODEL_PATH = "AIsumit123/mt5-base-nepali-gec-stage2"
mt5_prefix = "Correct sentence: "

MBART_MODEL_PATH = "tuyal/Stage2FFTmBart"
mbart_prefix = ""

NLLB_MODEL_PATH = "tuyal/Stage2FFTnllb200"
nllb_prefix = ""

MT5_RERANKER_PATH = "AIsumit123/muril-nepali-gec-reranker-mt5"
MBART_RERANKER_PATH = "AIsumit123/muril-reranker-mbart"
NLLB_RERANKER_PATH = "tuyal/nllbReranker"

# Configuration for available model choices
MODELS_CONFIG = {
    "mt5": {
        "model_path": MT5_MODEL_PATH,
        "prefix": mt5_prefix,
        "rerank_path": MT5_RERANKER_PATH,
        "tokenizer_type": "auto",
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

# In-memory cache for loaded models/tokenizers
LOADED = {}


def _unload_all_except(keep_choice=None):
    # Unload models/tokenizers not equal to keep_choice to free VRAM
    to_delete = [k for k in LOADED.keys() if k != keep_choice]
    for k in to_delete:
        entry = LOADED.pop(k)
        for name, obj in entry.items():
            try:
                del obj
            except Exception:
                pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _load_choice(choice):
    """Load models and tokenizers for a given choice into LOADED cache.
    This will unload any other loaded models to keep VRAM usage low.
    """
    if choice in LOADED:
        return LOADED[choice]

    if choice not in MODELS_CONFIG:
        raise ValueError(f"Unknown model choice: {choice}")

    # Unload others first
    _unload_all_except(keep_choice=None)

    cfg = MODELS_CONFIG[choice]

    # Load tokenizer
    if cfg["tokenizer_type"] == "mbart50":
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
    elif cfg["tokenizer_type"] == "nllb":
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])

    # special settings
    if choice == "mbart":
        tokenizer.src_lang = "ne_NP"
        tokenizer.tgt_lang = "ne_NP"
    if choice == "nllb":
        try:
            tokenizer.src_lang = "npi_Deva"
        except Exception:
            pass

    # Load seq2seq model
    seq_model = cfg["seq_class"].from_pretrained(cfg["model_path"]).to(device)
    seq_model.eval()

    # Load reranker tokenizer and model (trust_remote_code if needed)
    rerank_tokenizer = AutoTokenizer.from_pretrained(cfg["rerank_path"], trust_remote_code=True)
    rerank_model = cfg["rerank_class"].from_pretrained(cfg["rerank_path"], trust_remote_code=True).to(device)
    rerank_model.eval()

    entry = {
        "tokenizer": tokenizer,
        "prefix": cfg["prefix"],
        "model": seq_model,
        "rerank_tokenizer": rerank_tokenizer,
        "rerank_model": rerank_model,
    }

    LOADED[choice] = entry

    return entry

# =====================================================
# Generate candidate corrections
# =====================================================
def generate_candidates(input_text, model_choice="mt5", k=5):
    entry = _load_choice(model_choice)
    prefix = entry["prefix"]
    prefixed = prefix + input_text
    tokenizer = entry["tokenizer"]
    model = entry["model"]
    inputs = tokenizer(
        prefixed,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)
    # determine forced_bos_token_id when needed
    forced_bos_token_id = None
    if model_choice == "mbart":
        try:
            forced_bos_token_id = tokenizer.lang_code_to_id.get("ne_NP")
        except Exception:
            forced_bos_token_id = None
    elif model_choice == "nllb":
        try:
            forced_bos_token_id = tokenizer.convert_tokens_to_ids("npi_Deva")
        except Exception:
            forced_bos_token_id = None

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=5,
            num_return_sequences=5,
            max_new_tokens=inputs["input_ids"].shape[1] + 8,
            length_penalty=0.7,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            early_stopping=True,
            forced_bos_token_id=forced_bos_token_id,
        )

    candidates = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    # remove duplicates
    candidates = list(dict.fromkeys(candidates))

    return candidates


# =====================================================
# Pairwise comparison function
# =====================================================
def compare_pair(source, cand_A, cand_B, model_choice="mt5"):
    entry = _load_choice(model_choice)
    rerank_tokenizer = entry["rerank_tokenizer"]
    rerank_model = entry["rerank_model"]
    inputs = rerank_tokenizer(
        source,
        f"{cand_A} {rerank_tokenizer.sep_token} {cand_B}",
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        logits = rerank_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    # label 1 = A better
    prob_A = probs[0][1].item()

    return prob_A


# =====================================================
# Tournament reranking
# =====================================================
def rerank_candidates(source, candidates, model_choice="mt5"):

    n = len(candidates)
    scores = [0] * n

    for i in range(n):
        for j in range(i + 1, n):

            prob_A = compare_pair(source, candidates[i], candidates[j], model_choice)

            if prob_A > 0.5:
                scores[i] += 1
            else:
                scores[j] += 1

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


# =====================================================
# Predict best correction
# =====================================================
def predict_output(input_text, model_choice="mt5"):

    candidates = generate_candidates(input_text, model_choice=model_choice, k=5)

    ranked = rerank_candidates(input_text, candidates, model_choice)

    best_sentence = ranked[0][0]

    return best_sentence


# =====================================================
# Predict multiple sentences
# =====================================================
def predict_multiple(sentences, model_choice="mt5"):

    outputs = []

    for s in sentences:

        s = s.strip()

        if not s:
            outputs.append("")
            continue

        corrected = predict_output(s, model_choice)

        outputs.append(corrected)

    return outputs


# =====================================================
# Debug: show all candidates and scores
# =====================================================
def get_all_candidates_with_scores(input_text, model_choice="mt5"):

    candidates = generate_candidates(input_text, model_choice)

    ranked = rerank_candidates(input_text, candidates, model_choice)

    results = []

    for rank, (sent, score) in enumerate(ranked, 1):

        results.append({
            "rank": rank,
            "sentence": sent,
            "wins": score
        })

    return results
