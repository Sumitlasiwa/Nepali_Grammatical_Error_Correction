# Nepali GEC

Nepali GEC is a research and deployment codebase for Nepali grammatical error correction. It combines a sequence-to-sequence generator with an optional pairwise reranker, and also includes a FastAPI app for serving trained models.

The repository currently contains three main parts:

- `generator_training/`: train and evaluate the base correction model
- `reranker/`: build a pairwise reranker on top of generator candidates
- `deployment/`: serve correction models through an API and simple frontend

## Overview

The full workflow is:

1. Train or load a Nepali GEC generator model such as mT5.
2. Generate multiple correction candidates for each input sentence.
3. Train a MuRIL-based cross-encoder reranker to decide which candidate is best.
4. Serve the generator and reranker through the FastAPI app in `deployment/`.

The codebase is designed around Nepali-specific GEC experiments and uses:

- Hugging Face `transformers` and `datasets`
- Weights & Biases for experiment tracking
- Custom Nepali-oriented metrics including BLEU, GLEU, sentence exact match, and token-level F0.5
- FastAPI for inference serving

## Repository Structure

```text
nepali_gec/
|-- README.md
|-- requirements.txt
|-- .env
|-- data/                      # ignored in git; local datasets and generated files
|-- outputs/                   # ignored in git; checkpoints, logs, best models
|-- notebooks/                 # experiments and analysis notebooks
|-- tests/                     # sample test files, predictions, evaluation notebooks
|-- generator_training/
|   |-- trainer.py
|   |-- training_args.py
|   |-- config.py
|   |-- data_utils.py
|   |-- callbacks.py
|   |-- utils.py
|   `-- metrics/
|       |-- metrics.py
|       `-- compute_gleu.py
|-- reranker/
|   |-- generate_candidates.py
|   |-- build_pairwise_dataset.py
|   |-- train_reranker.py
|   |-- rerank_inference.py
|   `-- utils.py
`-- deployment/
    |-- app.py
    |-- requirements.txt
    |-- index.html
    |-- model/
    |   `-- predict.py
    `-- schema/
        `-- user_input_output.py
```

## Features

- Nepali grammar correction with transformer-based seq2seq models
- Optional LoRA / PEFT-based fine-tuning support
- Custom training callbacks for:
  - early stopping
  - sample prediction logging
  - CSV metric logging
  - periodic memory cleanup
  - mixed loss-only and full-metric evaluation
- Candidate generation + reranking pipeline
- FastAPI inference API with multiple model choices
- Evaluation notebooks and sample prediction files under `tests/`

## Models Used

The code currently references these external model and dataset identifiers:

- Generator training default model: `AIsumit123/mt5_base_v2_101`
- Generator training dataset: `AIsumit123/nepali_gec_data_v4`
- Deployment generator models:
  - `AIsumit123/mt5-base-nepali-gec-stage2`
  - `tuyal/Stage2FFTmBart`
  - `tuyal/Stage2FFTnllb200`
- Deployment rerankers:
  - `AIsumit123/muril-nepali-gec-reranker-mt5`
  - `AIsumit123/muril-reranker-mbart`
  - `tuyal/nllbReranker`

## Environment Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Nepali_Grammatical_Error_Correction
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

For the full research environment:

```bash
pip install -r requirements.txt
```

For deployment only:

```bash
pip install -r deployment/requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root with the keys you need:

```env
WANDB_KEY=your_wandb_api_key
HUGGINGFACE_KEY=your_huggingface_token
```

Notes:

- `WANDB_KEY` is needed for experiment tracking if you use Weights & Biases.
- `HUGGINGFACE_KEY` may be needed for gated or private model/dataset access.
- Do not commit `.env` to git.

## Data

The `data/` directory is intentionally ignored in git. Create it locally before running experiments.

Suggested layout:

```text
data/
|-- raw/
|-- cleaned/
|-- inflected/
|-- candidates.jsonl
|-- pairwise_train.jsonl
`-- predictions.jsonl
```

### Expected training dataset format

The generator training code expects a Hugging Face dataset with at least:

- `incorrect_sentence`
- `correct_sentence`

The sample file in `tests/test.json` also contains:

- `error_type`
- `edit`

Example record:

```json
{
  "incorrect_sentence": "म स्कुल जान्छ ।",
  "correct_sentence": "म स्कुल जान्छु ।",
  "error_type": "VERB:AGR"
}
```

## Generator Training

The generator training pipeline lives in `generator_training/`.

### Main files

- `config.py`: experiment configuration and hyperparameters
- `data_utils.py`: dataset loading, shuffling, sampling, tokenization
- `training_args.py`: model setup and Hugging Face training arguments
- `trainer.py`: main training entry point
- `callbacks.py`: custom callbacks and evaluation schedule
- `metrics/metrics.py`: BLEU, GLEU, exact match, token-level F0.5

### Default training configuration

Some notable defaults from `generator_training/config.py`:

- model: `AIsumit123/mt5_base_v2_101`
- dataset: `AIsumit123/nepali_gec_data_v4`
- max length: `64`
- batch size: `8`
- epochs: `8`
- gradient accumulation: `4`
- learning rate: `5e-6`
- validation subset for trainer eval loop: `680`
- W&B project: `nepali-grammar-correction`

### Metrics

The training code can compute:

- BLEU
- GLEU
- sentence exact match
- token-level precision / recall / F0.5

The F0.5 implementation is token-based rather than ERRANT-style edit-span evaluation, which makes it usable for Nepali without an ERRANT-equivalent annotation pipeline.

### Running training

The training code is organized around `generator_training/trainer.py`. A typical intended workflow is:

```bash
cd generator_training
python trainer.py
```

Before running large experiments, review and adjust:

- `generator_training/config.py`
- dataset identifier
- output directory
- W&B project/run name
- LoRA and precision settings

### Training outputs

Training creates:

- `outputs/checkpoints/`
- `outputs/best_model/`
- `outputs/logs/`
- CSV evaluation logs if enabled
- W&B run logs and sample prediction tables

## Reranker Pipeline

The reranker is a second-stage model that chooses the best correction from multiple generator outputs.

### Why reranking?

The generator may produce several valid-looking corrections, but not all of them are equally accurate. The reranker compares candidate pairs and learns which correction is better for a given source sentence.

### Pipeline steps

#### Step 1: Generate candidates

```bash
python reranker/generate_candidates.py \
  --model_path ./mt5-gec-finetuned \
  --input_file data/test.csv \
  --output_file data/candidates.jsonl
```

Supported input formats:

- CSV with columns `incorrect_sentence`, `correct_sentence`
- JSONL with fields `source`, `target`

Output format:

```json
{
  "source": "गलत वाक्य",
  "reference": "सही वाक्य",
  "candidates": [
    { "text": "candidate 1", "logprob": -2.13 },
    { "text": "candidate 2", "logprob": -2.89 }
  ]
}
```

#### Step 2: Build pairwise training data

```bash
python reranker/build_pairwise_dataset.py \
  --input_file data/candidates.jsonl \
  --output_file data/pairwise_train.jsonl
```

This script:

- scores candidates against the reference using GLEU
- creates ordered candidate pairs `(A, B)`
- assigns label `1` if `A` is better, otherwise `0`
- drops near-tie pairs to reduce noisy supervision

#### Step 3: Train the reranker

```bash
python reranker/train_reranker.py \
  --train_file data/pairwise_train.jsonl \
  --output_dir ./muril-reranker \
  --wandb_project nepali-gec-reranker \
  --wandb_run_name muril-run1
```

The reranker uses a MuRIL cross-encoder built with:

- source sentence
- candidate A
- candidate B

#### Step 4: Run reranked inference

Interactive mode:

```bash
python reranker/rerank_inference.py \
  --mt5_model ./mt5-gec-finetuned \
  --reranker_dir ./muril-reranker
```

Batch mode:

```bash
python reranker/rerank_inference.py \
  --mt5_model ./mt5-gec-finetuned \
  --reranker_dir ./muril-reranker \
  --input_file data/test.jsonl \
  --output_file data/predictions.jsonl
```

The final ranking can combine:

- reranker pairwise win rate
- generator log-probability

This hybrid score is enabled by default and can be disabled with `--no_hybrid`.

## API Deployment

The FastAPI app lives in `deployment/`.

### Start the server

From the `deployment/` directory:

```bash
cd deployment
uvicorn app:app --reload
```

Default endpoints:

- `GET /` - homepage message
- `GET /health` - service status and model version
- `POST /correct` - grammar correction endpoint

### Request format

```json
{
  "text": "म स्कुल जान्छ ।",
  "model": "mt5"
}
```

### Response format

```json
{
  "results": [
    {
      "input": "म स्कुल जान्छ ।",
      "best_output": "म स्कुल जान्छु ।",
      "all_candidates": [
        {
          "rank": 1,
          "sentence": "म स्कुल जान्छु ।",
          "wins": 4
        }
      ]
    }
  ]
}
```

### Supported deployment model choices

The deployment model registry currently includes:

- `mt5`
- `mbart`
- `nllb`

`mt5` is the default.

## Evaluation and Experiments

Useful files under `tests/`:

- `test.json`: sample Nepali correction dataset
- `predictions.txt`: saved text predictions
- `predictions.csv`: saved CSV predictions
- `evaluation.ipynb`: evaluation notebook
- `evaluation_mbart.ipynb`: MBART-specific evaluation notebook

These are useful for:

- quick qualitative inspection
- comparing model outputs
- notebook-based metric analysis

## Notes and Caveats

- This repository is under active restructuring. Older references to `src/` in previous documentation no longer match the current layout.
- Large directories such as `data/`, `outputs/`, `wandb/`, and `notebooks/` are git-ignored.
- Several scripts use local absolute-style imports, so the safest way to run them is from their own directory or exactly as shown above.
- GPU is strongly recommended for training and for loading the deployment models efficiently.
- The deployment app eagerly loads multiple large models on startup, so initial launch can be slow and memory-heavy.

## Recommended Workflow

If you are new to the project, this is the easiest order to follow:

1. Install dependencies and create `.env`.
2. Confirm access to the Hugging Face models and dataset.
3. Train or load a generator model.
4. Generate candidate corrections.
5. Build pairwise data and train the reranker.
6. Run batch or interactive reranked inference.
7. Start the FastAPI app for serving.

## Acknowledgment

This project is focused on improving grammatical error correction for Nepali, a comparatively low-resource language in NLP tooling. The design choices in this codebase reflect practical tradeoffs for low-resource GEC: transfer learning, reranking, and language-adapted evaluation.
