import os
import ast
import json
import torch
import pandas as pd
from collections import defaultdict
from .spans import extract_labeled_spans, normalize_bio
from .config import ID2LABEL

from typing import Dict, List, Any, Optional

EXPERIMENTS_DIR = "EXPERIMENTS"

def _detect_bio_violations(labels: List[str]) -> List[Dict[str, Any]]:
    """
    Mendeteksi pelanggaran urutan BIO pada sebuah label sequence.
    Mengembalikan daftar pelanggaran yang ditemukan sebelum normalisasi.
    """
    violations = []
    prev = "O"
    for i, lbl in enumerate(labels):
        if lbl == "I-SENTIMENT" and prev not in {"I-SENTIMENT", "B-SENTIMENT"}:
            violations.append({"index": i, "label": lbl, "prev_label": prev, "reason": "I-SENTIMENT tanpa B-SENTIMENT sebelumnya"})
        elif lbl == "I-ASPECT" and prev not in {"I-ASPECT", "B-ASPECT"}:
            violations.append({"index": i, "label": lbl, "prev_label": prev, "reason": "I-ASPECT tanpa B-ASPECT sebelumnya"})
        prev = lbl
    return violations

def compute_span_metrics(true_labels_list, pred_labels_list, normalized = False):
    """
    Computes exact-match span-level Precision, Recall, and F1 score:
    - Global micro-averaged across all entities
    - Per entity type (ASPECT, SENTIMENT)
    
    Adheres strictly to the specification in NLP-T10-02.
    """
    counts = {
        "GLOBAL": {"TP": 0, "FP": 0, "FN": 0},
        "ASPECT": {"TP": 0, "FP": 0, "FN": 0},
        "SENTIMENT": {"TP": 0, "FP": 0, "FN": 0}
    }

    for y_true, y_pred in zip(true_labels_list, pred_labels_list):
        true_spans = extract_labeled_spans(y_true, normalized)
        pred_spans = extract_labeled_spans(y_pred, normalized)

        # Matched spans (TP)
        matched = true_spans.intersection(pred_spans)
        
        # False positives: in pred but not in true
        fps = pred_spans - true_spans
        
        # False negatives: in true but not in pred
        fns = true_spans - pred_spans

        counts["GLOBAL"]["TP"] += len(matched)
        counts["GLOBAL"]["FP"] += len(fps)
        counts["GLOBAL"]["FN"] += len(fns)

        for ent, s, e in matched:
            if ent in counts:
                counts[ent]["TP"] += 1

        for ent, s, e in fps:
            if ent in counts:
                counts[ent]["FP"] += 1

        for ent, s, e in fns:
            if ent in counts:
                counts[ent]["FN"] += 1

    # Calculate P, R, F1
    results = {}
    for key, c in counts.items():
        tp = c["TP"]
        fp = c["FP"]
        fn = c["FN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results[key] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }

    return results

NEGATIONS_AND_INTENSIFIERS = {"tidak", "kurang", "belum", "bukan", "tanpa", "tak", "sangat", "banget", "sekali", "terlalu", "cukup", "lumayan", "agak"}

def categorize_errors(
    tokens_list: List[List[str]],
    y_true_list: List[List[str]],
    y_pred_list: List[List[str]],
    normalize: bool = False
) -> Dict[str, Any]:
    """
    Categorizes errors into 6 structured categories (NLP-T11):
    1. missed_aspect: Aspect in True, completely missing in Pred
    2. missed_opinion: Sentiment in True, completely missing in Pred
    3. aspect_sentiment_confusion: Entity type confusion at overlapping positions
    4. false_span: Span in Pred, completely absent in True
    5. boundary_error: Overlapping span of same type with differing start/end
    6. negation_modifier_error: Boundary or span error involving negations/intensifiers

    Parameter `normalize` mengontrol apakah normalize_bio diterapkan sebelum
    ekstraksi span. Jika True, pelanggaran urutan BIO akan diperbaiki terlebih
    dahulu. Jika False (default), label dipakai apa adanya.
    Pada kedua kasus, pelanggaran BIO yang ditemukan pada prediksi (y_pred)
    sebelum normalisasi akan dicatat di kunci "bio_violations".
    """
    error_counts = {
        "missed_aspect": 0,
        "missed_opinion": 0,
        "aspect_sentiment_confusion": 0,
        "false_span": 0,
        "boundary_error": 0,
        "negation_modifier_error": 0
    }

    examples_per_category = defaultdict(list)

    # --- Catat pelanggaran BIO sebelum normalisasi (selalu dilakukan) ---
    bio_violation_counts = {"true": 0, "pred": 0}
    bio_violation_examples = {"true": [], "pred": []}

    for sent_idx, (tokens, y_t, y_p) in enumerate(zip(tokens_list, y_true_list, y_pred_list)):
        violations_true = _detect_bio_violations(list(y_t))
        violations_pred = _detect_bio_violations(list(y_p))

        if violations_true:
            bio_violation_counts["true"] += len(violations_true)
            if len(bio_violation_examples["true"]) < 5:
                bio_violation_examples["true"].append({
                    "sent_idx": sent_idx,
                    "sentence": " ".join(tokens),
                    "violations": violations_true
                })
        if violations_pred:
            bio_violation_counts["pred"] += len(violations_pred)
            if len(bio_violation_examples["pred"]) < 5:
                bio_violation_examples["pred"].append({
                    "sent_idx": sent_idx,
                    "sentence": " ".join(tokens),
                    "violations": violations_pred
                })

    # --- Hitung span error utama dengan atau tanpa normalisasi ---
    for sent_idx, (tokens, y_t, y_p) in enumerate(zip(tokens_list, y_true_list, y_pred_list)):
        true_spans = extract_labeled_spans(y_t, normalize=normalize)
        pred_spans = extract_labeled_spans(y_p, normalize=normalize)

        # Matched spans
        matched = true_spans.intersection(pred_spans)
        unmatched_true = list(true_spans - matched)
        unmatched_pred = list(pred_spans - matched)

        handled_true = set()
        handled_pred = set()

        # Check boundary errors and confusion between overlapping spans
        for t_ent, t_s, t_e in unmatched_true:
            for p_ent, p_s, p_e in unmatched_pred:
                # Check token overlap
                if max(t_s, p_s) <= min(t_e, p_e):
                    # There is overlap!
                    overlap_tokens = tokens[max(t_s, p_s):min(t_e, p_e) + 1]
                    has_marker = any(tok.lower() in NEGATIONS_AND_INTENSIFIERS for tok in tokens[min(t_s, p_s):max(t_e, p_e) + 1])

                    if t_ent != p_ent:
                        error_counts["aspect_sentiment_confusion"] += 1
                        examples_per_category["aspect_sentiment_confusion"].append({
                            "sent_idx": sent_idx,
                            "sentence": " ".join(tokens),
                            "true_span": f"{t_ent} ('{' '.join(tokens[t_s:t_e+1])}')",
                            "pred_span": f"{p_ent} ('{' '.join(tokens[p_s:p_e+1])}')"
                        })
                    else:
                        # Same entity, boundary error
                        if has_marker:
                            error_counts["negation_modifier_error"] += 1
                            examples_per_category["negation_modifier_error"].append({
                                "sent_idx": sent_idx,
                                "sentence": " ".join(tokens),
                                "true_span": f"{t_ent} ('{' '.join(tokens[t_s:t_e+1])}')",
                                "pred_span": f"{p_ent} ('{' '.join(tokens[p_s:p_e+1])}')",
                                "note": "Melibatkan penanda negasi / penguat opini"
                            })
                        else:
                            error_counts["boundary_error"] += 1
                            examples_per_category["boundary_error"].append({
                                "sent_idx": sent_idx,
                                "sentence": " ".join(tokens),
                                "true_span": f"{t_ent} ('{' '.join(tokens[t_s:t_e+1])}')",
                                "pred_span": f"{p_ent} ('{' '.join(tokens[p_s:p_e+1])}')"
                            })

                    handled_true.add((t_ent, t_s, t_e))
                    handled_pred.add((p_ent, p_s, p_e))

        # Check completely missed spans
        for t_ent, t_s, t_e in unmatched_true:
            if (t_ent, t_s, t_e) not in handled_true:
                span_text = " ".join(tokens[t_s:t_e+1])
                if t_ent == "ASPECT":
                    error_counts["missed_aspect"] += 1
                    examples_per_category["missed_aspect"].append({
                        "sent_idx": sent_idx,
                        "sentence": " ".join(tokens),
                        "missed_span": f"ASPECT ('{span_text}')"
                    })
                elif t_ent == "SENTIMENT":
                    error_counts["missed_opinion"] += 1
                    examples_per_category["missed_opinion"].append({
                        "sent_idx": sent_idx,
                        "sentence": " ".join(tokens),
                        "missed_span": f"SENTIMENT ('{span_text}')"
                    })

        # Check completely false spans
        for p_ent, p_s, p_e in unmatched_pred:
            if (p_ent, p_s, p_e) not in handled_pred:
                span_text = " ".join(tokens[p_s:p_e+1])
                error_counts["false_span"] += 1
                examples_per_category["false_span"].append({
                    "sent_idx": sent_idx,
                    "sentence": " ".join(tokens),
                    "false_span": f"{p_ent} ('{span_text}')"
                })

    return {
        "normalize_bio_applied": normalize,
        "bio_violations": {
            "counts": bio_violation_counts,
            "sample_examples": bio_violation_examples
        },
        "error_counts": error_counts,
        "sample_examples": {k: v[:5] for k, v in examples_per_category.items()}
    }

def evaluate_crf_metrics(test_data_folder="data", model=None):
    from .train_crf import predict_test_crf  # lazy import to avoid circular dependency

    summary_path = os.path.join(EXPERIMENTS_DIR, "crf_final_summary.json")
    if not os.path.exists(summary_path):
        print("Summary hasil validasi tidak ditemukan, coba cek dan lakukan training ulang untuk model crf")
        return

    test_result = predict_test_crf(test_data_folder=test_data_folder, model=model)

    # Parse columns from CSV strings if needed
    if isinstance(test_result["tokens"].iloc[0], str):
        test_result["tokens"] = test_result["tokens"].apply(ast.literal_eval)
    if isinstance(test_result["seq_label_fixed"].iloc[0], str):
        test_result["seq_label_fixed"] = test_result["seq_label_fixed"].apply(ast.literal_eval)
    if isinstance(test_result["crf_pred"].iloc[0], str):
        test_result["crf_pred"] = test_result["crf_pred"].apply(ast.literal_eval)

    # Map seq_label_fixed (integer IDs) → string labels via ID2LABEL
    seq_label_mapped = [
        [ID2LABEL[label_id] for label_id in seq]
        for seq in test_result["seq_label_fixed"].tolist()
    ]

    # Load existing JSON
    with open(os.path.join(summary_path), "r") as f:
        data = json.load(f)
    
    metrics_raw = compute_span_metrics(seq_label_mapped, test_result['crf_pred'])
    
    record_raw = {
        "val_micro_f1": round(metrics_raw["GLOBAL"]["F1"], 4),
        "val_aspect_f1": round(metrics_raw["ASPECT"]["F1"], 4),
        "val_sentiment_f1": round(metrics_raw["SENTIMENT"]["F1"], 4),
        "val_metrics": metrics_raw
    }

    metrics_normalized = compute_span_metrics(seq_label_mapped, test_result['crf_pred'], normalized=True)
    
    record_normalized = {
        "val_micro_f1": round(metrics_normalized["GLOBAL"]["F1"], 4),
        "val_aspect_f1": round(metrics_normalized["ASPECT"]["F1"], 4),
        "val_sentiment_f1": round(metrics_normalized["SENTIMENT"]["F1"], 4),
        "val_metrics": metrics_normalized
    }

    # Add it under a new key
    data["test_eval_raw"] = record_raw
    data["test_eval_normalized"] = record_normalized 

    # Write back
    with open(os.path.join(summary_path), "w") as f:
        json.dump(data, f, indent=2)

    print(f"Analisa hasil CRF dapat dilihat di {summary_path}")

def analyze_crf_errors(test_data_folder="data", model=None, normalize: bool = False):
    from .train_crf import predict_test_crf  # lazy import to avoid circular dependency
    test_result = predict_test_crf(test_data_folder=test_data_folder, model=model)

    # Parse columns from CSV strings if needed
    if isinstance(test_result["tokens"].iloc[0], str):
        test_result["tokens"] = test_result["tokens"].apply(ast.literal_eval)
    if isinstance(test_result["seq_label_fixed"].iloc[0], str):
        test_result["seq_label_fixed"] = test_result["seq_label_fixed"].apply(ast.literal_eval)
    if isinstance(test_result["crf_pred"].iloc[0], str):
        test_result["crf_pred"] = test_result["crf_pred"].apply(ast.literal_eval)

    # Map seq_label_fixed (integer IDs) → string labels via ID2LABEL
    seq_label_mapped = [
        [ID2LABEL[label_id] for label_id in seq]
        for seq in test_result["seq_label_fixed"].tolist()
    ]

    crf_errors = categorize_errors(
        test_result["tokens"].tolist(),
        seq_label_mapped,
        test_result["crf_pred"].tolist(),
        normalize=normalize
    )

    # Save crf_errors to JSON
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    suffix = "_normalized" if normalize else "_raw"
    output_path = os.path.join(EXPERIMENTS_DIR, f"crf_error_analysis{suffix}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(crf_errors, f, indent=2, ensure_ascii=False)
    print(f"[analyze_crf_errors] Error analysis (normalize_bio={normalize}) saved to: {output_path}")

    return crf_errors

def evaluate_transformer_metrics(test_data_folder="data", model=None):
    from .train_transformer import predict_test_transformer  # lazy import to avoid circular dependency

    summary_path = os.path.join(EXPERIMENTS_DIR, "transformer_final_summary.json")
    if not os.path.exists(summary_path):
        print(f"Summary hasil validasi tidak ditemukan, coba cek di {summary_path}")
        return

    test_result = predict_test_transformer(test_data_folder=test_data_folder, model=model)

    # Parse columns from CSV strings if needed
    if isinstance(test_result["tokens"].iloc[0], str):
        test_result["tokens"] = test_result["tokens"].apply(ast.literal_eval)
    if isinstance(test_result["seq_label_fixed"].iloc[0], str):
        test_result["seq_label_fixed"] = test_result["seq_label_fixed"].apply(ast.literal_eval)
    if isinstance(test_result["transformer_pred"].iloc[0], str):
        test_result["transformer_pred"] = test_result["transformer_pred"].apply(ast.literal_eval)

    # Map seq_label_fixed (integer IDs) → string labels via ID2LABEL
    seq_label_mapped = [
        [ID2LABEL[label_id] for label_id in seq]
        for seq in test_result["seq_label_fixed"].tolist()
    ]

    # Load existing JSON
    with open(os.path.join(summary_path), "r") as f:
        data = json.load(f)
    
    metrics_raw = compute_span_metrics(seq_label_mapped, test_result["transformer_pred"])
    
    record_raw = {
        "val_micro_f1": round(metrics_raw["GLOBAL"]["F1"], 4),
        "val_aspect_f1": round(metrics_raw["ASPECT"]["F1"], 4),
        "val_sentiment_f1": round(metrics_raw["SENTIMENT"]["F1"], 4),
        "val_metrics": metrics_raw
    }

    metrics_normalized = compute_span_metrics(seq_label_mapped, test_result["transformer_pred"], normalized=True)
    
    record_normalized = {
        "val_micro_f1": round(metrics_normalized["GLOBAL"]["F1"], 4),
        "val_aspect_f1": round(metrics_normalized["ASPECT"]["F1"], 4),
        "val_sentiment_f1": round(metrics_normalized["SENTIMENT"]["F1"], 4),
        "val_metrics": metrics_normalized
    }

    # Add it under a new key
    data["test_eval_raw"] = record_raw
    data["test_eval_normalized"] = record_normalized 

    # Write back
    with open(os.path.join(summary_path), "w") as f:
        json.dump(data, f, indent=2)

    print(f"Analisa hasil transformer dapat dilihat di {summary_path}")
    

def analyze_transformer_errors(test_data_folder="data", model=None, normalize: bool = False):
    from .train_transformer import predict_test_transformer  # lazy import to avoid circular dependency
    test_result = predict_test_transformer(test_data_folder=test_data_folder, model=model)

    # Parse columns from CSV strings if needed
    if isinstance(test_result["tokens"].iloc[0], str):
        test_result["tokens"] = test_result["tokens"].apply(ast.literal_eval)
    if isinstance(test_result["seq_label_fixed"].iloc[0], str):
        test_result["seq_label_fixed"] = test_result["seq_label_fixed"].apply(ast.literal_eval)
    if isinstance(test_result["transformer_pred"].iloc[0], str):
        test_result["transformer_pred"] = test_result["transformer_pred"].apply(ast.literal_eval)

    # Map seq_label_fixed (integer IDs) → string labels via ID2LABEL
    seq_label_mapped = [
        [ID2LABEL[label_id] for label_id in seq]
        for seq in test_result["seq_label_fixed"].tolist()
    ]

    transformer_errors = categorize_errors(
        test_result["tokens"].tolist(),
        seq_label_mapped,
        test_result["transformer_pred"].tolist(),
        normalize=normalize
    )

    # Save transformer_errors to JSON
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    suffix = "_normalized" if normalize else "_raw"
    output_path = os.path.join(EXPERIMENTS_DIR, f"transformer_error_analysis{suffix}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transformer_errors, f, indent=2, ensure_ascii=False)
    print(f"[analyze_transformer_errors] Error analysis (normalize_bio={normalize}) saved to: {output_path}")
    
    return transformer_errors
