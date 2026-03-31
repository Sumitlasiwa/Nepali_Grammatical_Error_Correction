"""
Evaluation metrics for Nepali GEC with enable/disable support
"""

import evaluate
import numpy as np
import tempfile
import os
from typing import List
from .compute_gleu import GLEU

def compute_gleu_score(sources: List[str], predictions: List[str], references: List[str], 
                       n: int = 4, num_iterations: int = 500) -> float:
    """
    Compute corpus-level GLEU score using file-based approach.
    
    Args:
        sources: List of source (incorrect) sentences
        predictions: List of predicted (corrected) sentences  
        references: List of reference (correct) sentences
        n: n-gram order (default: 4)
        num_iterations: number of GLEU iterations (default: 500)
    
    Returns:
        GLEU score (0-100)
    """
    def write_temp_file(sentences: List[str]) -> str:
        """Write sentences to temporary file."""
        fd, path = tempfile.mkstemp(suffix='.txt')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                for sent in sentences:
                    f.write(sent.strip() + '\n')
        except Exception as e:
            os.close(fd)
            raise e
        return path
    
    # Initialize file paths to None for proper cleanup
    src_file = None
    hyp_file = None
    ref_file = None
    
    try:
        # Write to temporary files
        src_file = write_temp_file(sources)
        hyp_file = write_temp_file(predictions)
        ref_file = write_temp_file(references)
        
        # Compute GLEU
        gleu_calculator = GLEU(n)
        gleu_calculator.load_sources(src_file)
        gleu_calculator.load_references([ref_file])
        
        result = list(gleu_calculator.run_iterations(
            num_iterations=num_iterations,
            source=src_file,
            hypothesis=hyp_file,
            per_sent=False
        ))
        
        gleu_score = float(result[0][0]) * 100
        
    except Exception as e:
        print(f"⚠️ GLEU computation failed: {e}")
        import traceback
        traceback.print_exc()
        gleu_score = 0.0
        
    finally:
        # Clean up temporary files
        for f in [src_file, hyp_file, ref_file]:
            if f is not None and os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Warning: Could not remove temp file {f}: {e}")
    
    return gleu_score


# Standard F0.5 in GEC assumes you can define:
# Gold edits
# System edits
# Alignment between source > hypothesis > reference
# English GEC gets this from ERRANT.
# For Nepali, you do not have ERRANT, so you must change the unit of evaluation.

# Since no ERRANT-style evaluation framework exists for Nepali, we adopt a token-level edit-based F₀.₅ metric. Precision and recall are computed over token corrections rather than edit spans. While this differs from standard English GEC evaluation, it provides a transparent and language-agnostic measure commonly used in low-resource GEC.

# IMPLEMENTATION OF TOKEN-LEVEL EDIT-BASED F0.5
def token_f05(sources, preds, refs):
    tp = fp = fn = 0
    
    for src, pred, ref in zip(sources, preds, refs):
        src_t = src.split()
        pred_t = pred.split()
        ref_t = ref.split()

        L = min(len(src_t), len(pred_t), len(ref_t))

        for i in range(L):
            if pred_t[i] != src_t[i]:   # model made a change
                if pred_t[i] == ref_t[i]:
                    tp += 1 # correct edit
                else:
                    fp += 1 # wrong edit
                    
            if src_t[i] != ref_t[i] and pred_t[i] != ref_t[i]:
                fn += 1 # missed correction
                
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    beta2 = 0.5 ** 2
    f05 = (1 + beta2) * precision * recall / (beta2 * precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f05

# IMPLEMENTATION OF SENTENCE_EXACT_MATCH_ACCURACY
def normalize(sent):
    return " ".join(sent.strip().split())

def sentence_exact_match_accuracy(preds, refs):
    assert len(preds) == len(refs)
    
    correct = 0
    total = len(preds)

    for pred, ref in zip(preds, refs):
        if normalize(pred) == normalize(ref):
            correct += 1
            
    return correct / total if total > 0 else 0.0


def create_compute_metrics(tokenizer, config, eval_dataset):
    """
    Factory function to create compute_metrics with tokenizer and dataset.
    
    Args:
        tokenizer: HuggingFace tokenizer
        config: Config object with metric flags
        eval_dataset: Raw dataset with source sentences (for GLEU)
    
    Usage:
        compute_metrics = create_compute_metrics(tokenizer, config, eval_dataset)
    """
    # Get enabled metrics
    enabled = config.get_enabled_metrics()
    print(f"Enabled metrics:{', '.join(enabled)}")
        
    if config.bleu:
        bleu_metric = evaluate.load("sacrebleu")
        
    # Extract source sentences from dataset for GLEU
    if config.gleu:
        print(f"Extracting {len(eval_dataset)} source sentences for GLEU...")
        source_sentences = [item["incorrect_sentence"] for item in eval_dataset]
        print(f"Extracted {len(source_sentences)} source sentences")

    def compute_metrics(eval_pred):
        """
        Compute BLEU, chrF,GLEU, and Correction Accuracy for Nepali GEC.
        Handles both token IDs and plain text predictions.
        """
        predictions, labels = eval_pred

        # --- Handle tuple outputs (e.g., logits + labels) ---
        if isinstance(predictions, tuple):
            predictions = predictions[0]
            

        # else:
        # Convert to numpy arrays
        predictions = np.array(predictions)
        labels = np.array(labels)

        # Handle logits (vocab dimension)
        if predictions.ndim == 3:
            predictions = predictions.argmax(axis=-1)

        # Replace -100 with pad_token_id
        predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

        # Decode
        preds = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
           
        preds_clean = [p.strip() for p in preds]
        refs_clean = [r.strip() for r in refs]
        
        # For GLEU, also need source sentences (already extracted)
        if config.gleu:
            # Make sure we have the right number of sources
            srcs_clean = source_sentences[:len(preds_clean)]
            if len(srcs_clean) != len(preds_clean):
                print(f"⚠️ Source/prediction length mismatch: {len(srcs_clean)} vs {len(preds_clean)}")
        
        metrics = {}
             
        # -- GEC-GLEU--
        if config.gleu:
            try:  
        
                metrics["gleu"] = compute_gleu_score(
                    sources=srcs_clean,
                    references=refs_clean,
                    predictions=preds_clean,
                    n=4,
                    num_iterations=500
                )
            
            except Exception as e:
                print(f"GLEU computation failed: {e}")
                metrics["gleu"] = 0.0     
                
        # -- BLEU
        if config.bleu:
            try:
                non_empty_indices = [i for i, (p, r) in enumerate(zip(preds_clean, refs_clean)) if p and r]
                if non_empty_indices:
                    preds_bleu = [preds_clean[i] for i in non_empty_indices]
                    refs_bleu = [[refs_clean[i]] for i in non_empty_indices]
                    bleu_result = bleu_metric.compute(predictions=preds_bleu, references=refs_bleu)
                    metrics["bleu"] = bleu_result["score"]
                else:
                    metrics["bleu"] = 0.0
            except Exception as e:
                print(f"BLEU computation failed: {e}")
                metrics["bleu"] = 0.0

        # --- Correction Accuracy ---
        if config.Exact_Match:
            try:
                metrics["Exact Match"] = sentence_exact_match_accuracy(preds_clean, refs_clean)
            except Exception as e:
                print(f"Exact Match computation failed: {e}")
                metrics["Exact Match"] = 0.0
                
        # --- F0.5 ---
        if config.F05:
            try:
                metrics["precision"],metrics["recall"],metrics["F05"] = token_f05(srcs_clean,preds_clean, refs_clean)
            except Exception as e:
                print(f"F05 computation failed: {e}")
                metrics["F05"] = 0.0

            
        # --- Print one sample for sanity ---
        if len(preds_clean) > 0:
            print(f"🔍 Sample - Pred: '{preds_clean[0][:50]}...' | Ref: '{refs_clean[0][:50]}...' | Match: {preds_clean[0] == refs_clean[0]}")

        return metrics
    
    return compute_metrics


if __name__ == "__main__":
    # Test GLEU computation
    from config import Config
    from trainer import setup_model
    from datasets import Dataset
    
    
    config = Config()
    config.gleu = True
    model, tokenizer = setup_model(config)
    
    # Sample data
    sources = [
        "मेर नाम सन्तोष ह ।",
        "म स्कुल जान्छु ।"
    ]
    predictions = [
        "मेरो नाम सन्तोष हो ।",
        "म विद्यालय जान्छु ।"
    ]
    references = [
        "मेरो नाम सन्तोष हो ।",
        "म विद्यालय जान्छु ।"
    ]
    
    # Create a mock dataset with source sentences
    mock_dataset = Dataset.from_dict({
        "incorrect_sentence": sources,
        "correct_sentence": references
    })
    compute_metrics=create_compute_metrics(tokenizer, config, mock_dataset)
    # Tokenize predictions and references
    pred_ids = tokenizer(predictions, padding=True, truncation=True, max_length=64)["input_ids"]
    ref_ids = tokenizer(references, padding=True, truncation=True, max_length=64)["input_ids"]
    
    # Convert to numpy arrays (like Trainer does)
    pred_ids = np.array(pred_ids)
    ref_ids = np.array(ref_ids)
    
    # Create EvalPrediction-like tuple
    eval_pred = (pred_ids, ref_ids)
    metrics = compute_metrics(eval_pred)
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
