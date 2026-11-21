import os
import ast
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

from .evaluate import compute_span_metrics
from .transformer_data import create_tokenizer, tokenize_and_align_labels, align_predictions_back_to_words
from .config import (
    ID2LABEL, LABEL2ID, LABEL_LIST,
    MODEL_NAME, FALLBACK_MODEL_NAME,
    MAX_LEN, BATCH_SIZE,
    LEARNING_RATE, EPOCHS, WEIGHT_DECAY,
    set_seed, SEED
)

EXPERIMENTS_DIR = "EXPERIMENTS"
MODEL_SAVE_NAME = "transformer_best_model"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NERDataset(Dataset):
    """PyTorch Dataset untuk token-classification NER."""

    def __init__(self, df, tokenizer):
        self.samples = []

        for _, row in df.iterrows():
            tokens    = row["tokens"]
            seq_label = row["seq_label"]

            # seq_label berisi ID integer; ubah ke string label agar sesuai
            # dengan tokenize_and_align_labels yang menerima string label
            str_labels = [ID2LABEL[l] for l in seq_label]

            encoded = tokenize_and_align_labels(tokens, str_labels, tokenizer)
            self.samples.append(encoded)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer):
    """Padding dinamis pada setiap batch."""
    input_ids      = [item["input_ids"]      for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels         = [item["labels"]         for item in batch]

    # Pad ke panjang maksimum dalam batch
    max_len = max(len(x) for x in input_ids)

    pad_id    = tokenizer.pad_token_id
    pad_label = -100

    input_ids_padded      = []
    attention_mask_padded = []
    labels_padded         = []

    for ids, mask, lbl in zip(input_ids, attention_mask, labels):
        pad_len = max_len - len(ids)
        input_ids_padded.append(ids + [pad_id]    * pad_len)
        attention_mask_padded.append(mask + [0]   * pad_len)
        labels_padded.append(lbl + [pad_label]    * pad_len)

    return {
        "input_ids":      torch.tensor(input_ids_padded,      dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_padded, dtype=torch.long),
        "labels":         torch.tensor(labels_padded,         dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_transformer_data(df):
    """Parse kolom tokens & seq_label dari string (CSV) ke list Python."""
    for col in ("tokens", "seq_label"):
        if isinstance(df[col].iloc[0], str):
            df[col] = df[col].apply(ast.literal_eval)
    if "seq_label_fixed" in df.columns and isinstance(df["seq_label_fixed"].iloc[0], str):
        df["seq_label_fixed"] = df["seq_label_fixed"].apply(ast.literal_eval)
    return df


# ---------------------------------------------------------------------------
# Single epoch train / eval helpers
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, optimizer=None, scheduler=None, device="cpu"):
    """Jalankan satu epoch (train jika optimizer diberikan, else eval)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_true, all_pred = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss   = outputs.loss
            logits = outputs.logits  # (B, T, num_labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()

            # Decode predictions per sequence (hanya posisi bukan -100)
            preds  = logits.argmax(dim=-1).cpu().numpy()   # (B, T)
            lbls   = labels.cpu().numpy()                  # (B, T)
            masks  = attention_mask.cpu().numpy()          # (B, T)
            # word_ids tidak disimpan di batch; kita pakai pelabelan first-subword
            # (tanda: label != -100 & mask == 1) untuk rekonstruksi label asli
            for pred_seq, lbl_seq, mask_seq in zip(preds, lbls, masks):
                true_word, pred_word = [], []
                for p, l, m in zip(pred_seq, lbl_seq, mask_seq):
                    if m == 0:
                        break
                    if l == -100:
                        continue   # skip special / continuation subword
                    true_word.append(ID2LABEL.get(int(l), "O"))
                    pred_word.append(ID2LABEL.get(int(p), "O"))
                all_true.append(true_word)
                all_pred.append(pred_word)

    avg_loss = total_loss / len(loader)
    metrics  = compute_span_metrics(all_true, all_pred)
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Load saved model
# ---------------------------------------------------------------------------

def load_best_model(model_dir=None, device="cpu"):
    """Muat model Transformer terbaik dari disk."""
    if model_dir is None:
        model_dir = os.path.join(EXPERIMENTS_DIR, MODEL_SAVE_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        model_dir,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_transformer(df_train, df_val):
    """
    Melatih model Transformer untuk NER dengan grid search hyperparameter
    pada validation set, mengikuti alur yang sama dengan train_crf.

    Returns:
        model: Model Transformer terbaik.
    """
    set_seed(SEED)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    model_dir    = os.path.join(EXPERIMENTS_DIR, MODEL_SAVE_NAME)
    summary_path = os.path.join(EXPERIMENTS_DIR, "transformer_final_summary.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load jika model sudah ada
    # ------------------------------------------------------------------
    model_exists   = os.path.isdir(model_dir) and os.listdir(model_dir)
    summary_exists = os.path.exists(summary_path)

    if model_exists and summary_exists:
        print(f"Trained Transformer model loaded: {model_dir}")

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        print(f"[loaded] Summary: {summary_path}")
        print(summary)

        model = load_best_model(model_dir, device=device)
        return model

    # ------------------------------------------------------------------
    # Parse data
    # ------------------------------------------------------------------
    for df in (df_train, df_val):
        prepare_transformer_data(df)

    print(f"Jumlah data — Train: {len(df_train)}, Val: {len(df_val)}")

    # ------------------------------------------------------------------
    # Tokenizer (coba MODEL_NAME, fallback ke FALLBACK_MODEL_NAME)
    # ------------------------------------------------------------------
    try:
        tokenizer = create_tokenizer(MODEL_NAME)
        base_model_name = MODEL_NAME
        print(f"Tokenizer loaded: {MODEL_NAME}")
    except Exception as e:
        print(f"Gagal memuat {MODEL_NAME}: {e}\nMencoba fallback: {FALLBACK_MODEL_NAME}")
        tokenizer = create_tokenizer(FALLBACK_MODEL_NAME)
        base_model_name = FALLBACK_MODEL_NAME
        print(f"Tokenizer loaded (fallback): {FALLBACK_MODEL_NAME}")

    # ------------------------------------------------------------------
    # Build dataset & DataLoader (dilakukan sekali, dibagi per run)
    # ------------------------------------------------------------------
    print("Tokenisasi dan alignment label ...")
    train_dataset = NERDataset(df_train, tokenizer)
    val_dataset   = NERDataset(df_val,   tokenizer)

    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            collate_fn=lambda b: collate_fn(b, tokenizer),
        )

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader   = make_loader(val_dataset,   shuffle=False)

    # ------------------------------------------------------------------
    # Grid Search Hyperparameter
    # ------------------------------------------------------------------
    tuning_results  = []
    best_micro_f1   = -1.0
    best_params     = None
    best_model_state = None

    print(f"\n--- Grid Search Hyperparameter Transformer pada Validation Set ---")
    print(f"Ruang parameter: lr in {LEARNING_RATE}, weight_decay in {WEIGHT_DECAY}, epochs={EPOCHS}")

    for lr in LEARNING_RATE:
        for wd in WEIGHT_DECAY:
            print(f"\n>> Melatih dengan lr={lr}, weight_decay={wd} ...")

            # Inisialisasi model baru untuk setiap kombinasi HP
            try:
                model = AutoModelForTokenClassification.from_pretrained(
                    base_model_name,
                    num_labels=len(LABEL_LIST),
                    id2label=ID2LABEL,
                    label2id=LABEL2ID,
                    ignore_mismatched_sizes=True,
                )
            except Exception:
                model = AutoModelForTokenClassification.from_pretrained(
                    FALLBACK_MODEL_NAME,
                    num_labels=len(LABEL_LIST),
                    id2label=ID2LABEL,
                    label2id=LABEL2ID,
                    ignore_mismatched_sizes=True,
                )
            model.to(device)

            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
            total_steps = len(train_loader) * EPOCHS
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps,
            )

            # Training loop
            for epoch in range(1, EPOCHS + 1):
                train_loss, train_metrics = _run_epoch(
                    model, train_loader, optimizer, scheduler, device
                )
                print(
                    f"  Epoch {epoch}/{EPOCHS} — "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Micro-F1: {train_metrics['GLOBAL']['F1']:.4f}"
                )

            # Evaluasi pada validation set
            val_loss, val_metrics = _run_epoch(model, val_loader, device=device)
            micro_f1 = val_metrics["GLOBAL"]["F1"]

            record = {
                "lr":               lr,
                "weight_decay":     wd,
                "epochs":           EPOCHS,
                "val_loss":         round(val_loss,   4),
                "val_micro_f1":     round(micro_f1,   4),
                "val_aspect_f1":    round(val_metrics["ASPECT"]["F1"],     4),
                "val_sentiment_f1": round(val_metrics["SENTIMENT"]["F1"],  4),
                "val_metrics":      val_metrics,
            }
            tuning_results.append(record)

            print(
                f"lr={lr} | wd={wd} -> "
                f"Val Micro-F1: {micro_f1:.4f} "
                f"(ASPECT: {val_metrics['ASPECT']['F1']:.4f}, "
                f"SENTIMENT: {val_metrics['SENTIMENT']['F1']:.4f})"
            )

            if micro_f1 > best_micro_f1:
                best_micro_f1    = micro_f1
                best_params      = {"lr": lr, "weight_decay": wd, "epochs": EPOCHS}
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_tokenizer   = tokenizer

    print(f"\nKonfigurasi Transformer Terbaik: {best_params} dengan Val Micro-F1 = {best_micro_f1:.4f}")

    # ------------------------------------------------------------------
    # Simpan hasil tuning
    # ------------------------------------------------------------------
    tuning_log_path = os.path.join(EXPERIMENTS_DIR, "transformer_tuning_results.json")
    with open(tuning_log_path, "w", encoding="utf-8") as f:
        json.dump(tuning_results, f, indent=2)
    print(f"Hasil tuning disimpan ke: {tuning_log_path}")

    # ------------------------------------------------------------------
    # Simpan model terbaik (muat ulang state dict)
    # ------------------------------------------------------------------
    try:
        best_model = AutoModelForTokenClassification.from_pretrained(
            base_model_name,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    except Exception:
        best_model = AutoModelForTokenClassification.from_pretrained(
            FALLBACK_MODEL_NAME,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

    best_model.load_state_dict(best_model_state)
    best_model.to(device)

    os.makedirs(model_dir, exist_ok=True)
    best_model.save_pretrained(model_dir)
    best_tokenizer.save_pretrained(model_dir)
    print(f"Model Transformer terbaik disimpan ke: {model_dir}")

    # ------------------------------------------------------------------
    # Evaluasi final pada validation set
    # ------------------------------------------------------------------
    val_loss_final, val_eval_final = _run_epoch(best_model, val_loader, device=device)

    print("\n=== Evaluasi Final Transformer pada Validation ===")
    print(
        f"Validation Micro-F1 : {val_eval_final['GLOBAL']['F1']:.4f} "
        f"(P: {val_eval_final['GLOBAL']['Precision']:.4f}, "
        f"R: {val_eval_final['GLOBAL']['Recall']:.4f})"
    )
    print(
        f"  - ASPECT    : "
        f"P={val_eval_final['ASPECT']['Precision']:.4f}, "
        f"R={val_eval_final['ASPECT']['Recall']:.4f}, "
        f"F1={val_eval_final['ASPECT']['F1']:.4f}"
    )
    print(
        f"  - SENTIMENT : "
        f"P={val_eval_final['SENTIMENT']['Precision']:.4f}, "
        f"R={val_eval_final['SENTIMENT']['Recall']:.4f}, "
        f"F1={val_eval_final['SENTIMENT']['F1']:.4f}"
    )

    summary = {
        "best_params":     best_params,
        "validation_eval": val_eval_final,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(summary)

    return best_model

def predict_test_transformer(test_data_folder="data", model=None):
    test_csv_path           = os.path.join(test_data_folder, "test.csv")
    test_predicted_csv_path = os.path.join(test_data_folder, "test_predicted.csv")

    # Cek apakah test_predicted.csv sudah ada; jika ada, load dengan pandas
    if os.path.exists(test_predicted_csv_path):
        test_predicted_csv = pd.read_csv(test_predicted_csv_path)

        # Jika kolom transformer_pred sudah ada, langsung return
        if "transformer_pred" in test_predicted_csv.columns:
            print("[predict_test_transformer] Prediksi Transformer sudah ada, melewati inferensi.")
            return test_predicted_csv

    else:
        # Copy test.csv, simpan sebagai test_predicted.csv
        test_predicted_csv = pd.read_csv(test_csv_path)

    # Jika model tidak diberikan, load best Transformer model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = load_best_model(device=device)

    model.eval()

    # Load tokenizer dari direktori model yang tersimpan
    model_dir = model.config._name_or_path if hasattr(model.config, "_name_or_path") else "EXPERIMENTS/transformer_best_model"
    tokenizer = create_tokenizer(model_dir)

    # Parse kolom tokens jika masih berupa string
    if isinstance(test_predicted_csv["tokens"].iloc[0], str):
        test_predicted_csv["tokens"] = test_predicted_csv["tokens"].apply(ast.literal_eval)

    all_predictions = []

    with torch.no_grad():
        for tokens in test_predicted_csv["tokens"]:
            # Tokenisasi dengan tokenizer model
            encoded = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                return_tensors="pt"
            )

            word_ids = encoded.word_ids()

            input_ids      = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits  # (1, T, num_labels)

            subword_preds = logits.argmax(dim=-1)[0].cpu().tolist()

            # Align hasil prediksi kembali ke level kata
            word_preds = align_predictions_back_to_words(word_ids, subword_preds, ID2LABEL)
            all_predictions.append(word_preds)

    # Simpan hasil prediksi pada kolom transformer_pred
    test_predicted_csv["transformer_pred"] = all_predictions

    # Simpan ke file
    test_predicted_csv.to_csv(test_predicted_csv_path, index=False)
    print(f"[predict_test_transformer] Prediksi disimpan ke: {test_predicted_csv_path}")

    return test_predicted_csv