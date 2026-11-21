import pandas as pd
import os
from typing import Dict
from datasets import load_dataset
from .config import LABEL_LIST, DATASET_PATH, DATASET_NAME, ID2LABEL

def print_issues(split, df_len, issues):
    print(f"[{split.upper()}] Rows: {df_len}")
    print(f"  Empty sentences: {len(issues['empty_sentences'])}")
    print(f"  Length mismatches: {len(issues['mismatched_lengths'])}")
    print(f"  Invalid labels: {len(issues['invalid_labels'])}")
    print(f"  Illegal transitions: {len(issues['illegal_transitions'])}")

def load_and_filter(data_folder = "data"):
    """
    Validates the dataset annotations automatically (NLP-T2-03):
    - checks that token count == label count for every sentence
    - checks for illegal transitions (e.g. O -> I-X, B-ASPECT -> I-SENTIMENT)
    - checks for empty sentences
    """

    train_path = os.path.join(data_folder, "train.csv")
    test_path = os.path.join(data_folder, "test.csv")
    val_path = os.path.join(data_folder, "val.csv")

    paths_exist = all(map(os.path.exists, (train_path, test_path, val_path)))

    if paths_exist:
        train_df, test_df, val_df = (
            pd.read_csv(p) for p in (train_path, test_path, val_path)
        )

        return train_df, test_df, val_df

    raw_ds = load_dataset(DATASET_PATH, DATASET_NAME, trust_remote_code=True)
    
    dfs = {}

    issues = {
        "mismatched_lengths": [],
        "empty_sentences": [],
        "invalid_labels": [],
        "illegal_transitions": []
    }

    for split in ["train", "validation", "test"]:
        df = pd.DataFrame({
            "tokens": raw_ds[split]["tokens"],
            "seq_label": raw_ds[split]["seq_label"],
        })
        df['id'] = df.reset_index().index

        for idx, row in df.iterrows():
            tokens = list(row["tokens"])
            labels = list(row["seq_label"])
            sent_id = row["id"]

            # Check empty
            if len(tokens) == 0:
                issues["empty_sentences"].append(sent_id)
                continue

            # Check length match
            if len(tokens) != len(labels):
                issues["mismatched_lengths"].append({
                    "id": sent_id,
                    "n_tokens": len(tokens),
                    "n_labels": len(labels)
                })

            # Check illegal transitions
            prev_label = 1 # "O"
            for i, (tok, lbl) in enumerate(zip(tokens, labels)):
                # Transition check: e.g., I-ASPECT must follow B-ASPECT or I-ASPECT

                if lbl == 0 or lbl == 2:
                    expected_prev = [0, 3] if lbl == 0 else [2, 4] 
                    if prev_label not in expected_prev:
                        issues["illegal_transitions"].append({
                            "id": sent_id,
                            "token_idx": i,
                            "token": tok,
                            "prev_label": ID2LABEL[prev_label],
                            "curr_label": ID2LABEL[lbl]
                        })
                prev_label = lbl

        print_issues(split, len(df), issues)
        dfs[split] = df

    return dfs["train"], dfs["validation"], dfs["test"]
    # return issues

# def test():
#     ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
#     LABEL2ID = {label: i for i, label in ID2LABEL.items()}
#     print(ID2LABEL)
#     print(LABEL2ID)