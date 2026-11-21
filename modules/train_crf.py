import os
import ast
import joblib
import json
import pandas as pd
import sklearn_crfsuite
from .evaluate import compute_span_metrics
from .crf_features import sentence2features
from .config import ID2LABEL

from .config import (
    CRF_C1_VALUES, CRF_C2_VALUES, CRF_ALGORITHM, CRF_MAX_ITERATIONS
)

EXPERIMENTS_DIR = "EXPERIMENTS"

def prepare_crf_data(df):
    X = sentence2features(df["tokens"].tolist())
    y = [[ID2LABEL[label] for label in seq] for seq in df["seq_label"].tolist()]

    return X, y

def load_best_model(model_name="crf_best_model.joblib"):
    model_path = os.path.join(EXPERIMENTS_DIR, model_name)
    crf_model = joblib.load(model_path)

    return crf_model

def train_crf(df_train, df_val):
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    model_path   = os.path.join(EXPERIMENTS_DIR, "crf_best_model.joblib")
    summary_path = os.path.join(EXPERIMENTS_DIR, "crf_final_summary.json")

    model_exists   = os.path.exists(model_path)
    summary_exists = os.path.exists(summary_path)

    if model_exists and summary_exists:
        print(f"Trained CRF model loaded  : {model_path}")

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        print(f"[loaded] Summary: {summary_path}")
        print(summary)

        crf_model = load_best_model()

        return crf_model

    for df in (df_train, df_val):
        df["tokens"]    = df["tokens"].apply(ast.literal_eval)
        df["seq_label"] = df["seq_label"].apply(ast.literal_eval)
        # same for seq_label_fixed if you use it
        if "seq_label_fixed" in df.columns:
            df["seq_label_fixed"] = df["seq_label_fixed"].apply(ast.literal_eval)

    print(f"Ekstraksi fitur kalimat (Train: {len(df_train)}, Val: {len(df_val)} ...")
    X_train, y_train = prepare_crf_data(df_train)
    X_val, y_val = prepare_crf_data(df_val)

    tuning_results = []
    best_micro_f1 = -1.0
    best_params = None
    best_crf = None

    print(f"\n--- Grid Search Hyperparameter CRF pada Validation Set ---")
    print(f"Ruang parameter: c1 in {CRF_C1_VALUES}, c2 in {CRF_C2_VALUES}")

    for c1 in CRF_C1_VALUES:
        for c2 in CRF_C2_VALUES:
            crf = sklearn_crfsuite.CRF(
                algorithm=CRF_ALGORITHM,
                c1=c1,
                c2=c2,
                max_iterations=CRF_MAX_ITERATIONS,
                all_possible_transitions=True
            )
            crf.fit(X_train, y_train)

            # Prediksi validation
            y_val_pred = crf.predict(X_val)
            metrics = compute_span_metrics(y_val, y_val_pred)
            micro_f1 = metrics["GLOBAL"]["F1"]

            record = {
                "c1": c1,
                "c2": c2,
                "algorithm": CRF_ALGORITHM,
                "val_micro_f1": round(micro_f1, 4),
                "val_aspect_f1": round(metrics["ASPECT"]["F1"], 4),
                "val_sentiment_f1": round(metrics["SENTIMENT"]["F1"], 4),
                "val_metrics": metrics
            }
            tuning_results.append(record)

            print(f"c1={c1:<5} | c2={c2:<5} -> Val Micro-F1: {micro_f1:.4f} (ASPECT: {metrics['ASPECT']['F1']:.4f}, SENTIMENT: {metrics['SENTIMENT']['F1']:.4f})")

            if micro_f1 > best_micro_f1:
                best_micro_f1 = micro_f1
                best_params = {"c1": c1, "c2": c2}
                best_crf = crf

    print(f"\nKonfigurasi CRF Terbaik: {best_params} dengan Val Micro-F1 = {best_micro_f1:.4f}")

    # Simpan hasil tuning ke JSON (NLP-T8-01 DoD)
    tuning_log_path = os.path.join(EXPERIMENTS_DIR, "crf_tuning_results.json")
    with open(tuning_log_path, "w", encoding="utf-8") as f:
        json.dump(tuning_results, f, indent=2)

    # Simpan model terbaik
    joblib.dump(best_crf, model_path)
    print(f"Model CRF terbaik disimpan ke: {model_path}")

    # Prediksi validation & test dengan model terbaik (NLP-T8-02)
    y_val_pred_best = best_crf.predict(X_val)

    val_eval_final = compute_span_metrics(y_val, y_val_pred_best)

    print("\n=== Evaluasi Final CRF pada Validation ===")
    print(f"Validation Micro-F1 : {val_eval_final['GLOBAL']['F1']:.4f} (P: {val_eval_final['GLOBAL']['Precision']:.4f}, R: {val_eval_final['GLOBAL']['Recall']:.4f})")
    print(f"  - ASPECT    (Test): P={val_eval_final['ASPECT']['Precision']:.4f}, R={val_eval_final['ASPECT']['Recall']:.4f}, F1={val_eval_final['ASPECT']['F1']:.4f}")
    print(f"  - SENTIMENT (Test): P={val_eval_final['SENTIMENT']['Precision']:.4f}, R={val_eval_final['SENTIMENT']['Recall']:.4f}, F1={val_eval_final['SENTIMENT']['F1']:.4f}")

    summary = {
        "best_params": best_params,
        "validation_eval": val_eval_final,
    }
    with open(os.path.join(summary_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(summary)

    return best_crf

def predict_test_crf(test_data_folder="data", model=None):
    test_csv_path           = os.path.join(test_data_folder, "test.csv")
    test_predicted_csv_path = os.path.join(test_data_folder, "test_predicted.csv")

    # Cek apakah test_predicted.csv sudah ada; jika ada, load dengan pandas
    if os.path.exists(test_predicted_csv_path):
        test_predicted_csv = pd.read_csv(test_predicted_csv_path)

        # Jika kolom crf_pred sudah ada, langsung return
        if "crf_pred" in test_predicted_csv.columns:
            print("[predict_test_crf] Prediksi CRF sudah ada, melewati inferensi.")
            return test_predicted_csv

    else:
        # Copy test.csv, simpan sebagai test_predicted.csv
        test_predicted_csv = pd.read_csv(test_csv_path)

    # Jika model tidak diberikan, load best CRF model
    if model is None:
        model = load_best_model()

    # Parse kolom tokens jika masih berupa string
    if isinstance(test_predicted_csv["tokens"].iloc[0], str):
        test_predicted_csv["tokens"] = test_predicted_csv["tokens"].apply(ast.literal_eval)

    # Ekstraksi fitur dan prediksi
    X_test = sentence2features(test_predicted_csv["tokens"].tolist())
    predictions = model.predict(X_test)

    # Simpan hasil prediksi pada kolom crf_pred
    test_predicted_csv["crf_pred"] = predictions

    # Simpan ke file
    test_predicted_csv.to_csv(test_predicted_csv_path, index=False)
    print(f"[predict_test_crf] Prediksi disimpan ke: {test_predicted_csv_path}")

    return test_predicted_csv
