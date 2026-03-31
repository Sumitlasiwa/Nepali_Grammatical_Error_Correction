
MODEL_VERSION = '1.0.0'


import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    MBart50TokenizerFast
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# Generator model (mT5)
# ===============================
MT5_MODEL_PATH = "AIsumit123/mt5-base-nepali-gec-stage2"
mt5_prefix = "Correct sentence: "
mt5_tokenizer = AutoTokenizer.from_pretrained(MT5_MODEL_PATH)
mt5_model = AutoModelForSeq2SeqLM.from_pretrained(MT5_MODEL_PATH).to(device)
mt5_model.eval()

# ===============================
# Generator model (mbart)
# ===============================
MBART_MODEL_PATH = "tuyal/Stage2FFTmBart"
mbart_prefix = ""
mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
mbart_model = AutoModelForSeq2SeqLM.from_pretrained(MBART_MODEL_PATH).to(device)
mbart_tokenizer.src_lang = "ne_NP"
mbart_tokenizer.tgt_lang = "ne_NP"
mbart_model.eval()

# ===============================
# Generator model (nllb)
# ===============================
NLLB_MODEL_PATH = "tuyal/Stage2FFTnllb200"
nllb_prefix = ""
nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_PATH).to(device)
nllb_tokenizer.src_lang = "npi_Deva"
nllb_model.eval()

# ===============================
# Pairwise reranker model (mt5)
# ===============================
MT5_RERANKER_PATH = "AIsumit123/muril-nepali-gec-reranker-mt5"
mt5_rerank_tokenizer = AutoTokenizer.from_pretrained(MT5_RERANKER_PATH,trust_remote_code=True)
mt5_rerank_model = AutoModelForSequenceClassification.from_pretrained(
    MT5_RERANKER_PATH,trust_remote_code=True
).to(device)
mt5_rerank_model.eval()

# ===============================
# Pairwise reranker model (mbart)
# ===============================
MBART_RERANKER_PATH = "AIsumit123/muril-reranker-mbart"
mbart_rerank_tokenizer = AutoTokenizer.from_pretrained(MBART_RERANKER_PATH,trust_remote_code=True)
mbart_rerank_model = AutoModelForSequenceClassification.from_pretrained(
    MBART_RERANKER_PATH,trust_remote_code=True
).to(device)
mbart_rerank_model.eval()

# ===============================
# Pairwise reranker model (nllb)
# ===============================
NLLB_RERANKER_PATH = "tuyal/nllbReranker"
nllb_rerank_tokenizer = AutoTokenizer.from_pretrained(NLLB_RERANKER_PATH,trust_remote_code=True)
nllb_rerank_model = AutoModelForSequenceClassification.from_pretrained(
    NLLB_RERANKER_PATH,trust_remote_code=True
).to(device)
nllb_rerank_model.eval()


# ── Model registry ──────────────────────────────────────────
MODELS = {
    "mt5": {
        "tokenizer": mt5_tokenizer,
        "prefix": mt5_prefix,
        "model": mt5_model,
        "rerank_tokenizer": mt5_rerank_tokenizer,
        "rerank_model": mt5_rerank_model,
    },
    "mbart": {
        "tokenizer": mbart_tokenizer,
        "prefix": mbart_prefix,
        "model": mbart_model,
        "rerank_tokenizer": mbart_rerank_tokenizer,
        "rerank_model": mbart_rerank_model,
    },
    "nllb": {
        "tokenizer": nllb_tokenizer,
        "prefix": nllb_prefix,
        "model": mbart_model,
        "rerank_tokenizer": nllb_rerank_tokenizer,
        "rerank_model": nllb_rerank_model,
    },
}

# =====================================================
# Generate candidate corrections
# =====================================================
def generate_candidates(input_text, model_choice="mt5", k=5):

    prefix = MODELS[model_choice]["prefix"]
    prefixed = prefix + input_text
    tokenizer = MODELS[model_choice]["tokenizer"]
    model = MODELS[model_choice]["model"]
    inputs = tokenizer(
        prefixed,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

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
            
            forced_bos_token_id = (
                tokenizer.lang_code_to_id["ne_NP"] if model_choice == "mbart"
                else tokenizer.convert_tokens_to_ids("npi_Deva") if model_choice == "nllb"
                else None
            )
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
    rerank_tokenizer = MODELS[model_choice]["rerank_tokenizer"]
    rerank_model = MODELS[model_choice]["rerank_model"]
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

    candidates = generate_candidates(input_text, k=5)

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