import pandas as pd
import ast
from pathlib import Path
from modules.data_loader import load_and_filter
from modules.annotation import validate_annotation_manual
from modules.train_crf import train_crf
from modules.train_transformer import train_transformer
from modules.evaluate import analyze_crf_errors, analyze_transformer_errors, evaluate_crf_metrics, evaluate_transformer_metrics
from modules.transformer_data import demonstrate_alignment

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

train, val, test = load_and_filter()

splits = {"train": train, "val": val, "test": test}
paths = {name: OUT_DIR / f"{name}.csv" for name in splits}

all_exist = all(path.exists() for path in paths.values())

if all_exist:
    for name, path in paths.items():
        splits[name] = pd.read_csv(path)
        print(f"[loaded] Annotated data sudah tersedia: {name}")
else:
    for name, df in splits.items():
        path = paths[name]
        fixed = validate_annotation_manual(df)
        fixed.to_csv(path, index=False)
        splits[name] = fixed
        print(f"[saved] {name}: {len(fixed)} baris -> {path}")

train, val, test = splits["train"], splits["val"], splits["test"]

crf_model = train_crf(train, val)

evaluate_crf_metrics("data", crf_model)
crf_errors_raw = analyze_crf_errors(OUT_DIR, crf_model, normalize=False)
crf_errors_normalized = analyze_crf_errors(OUT_DIR, crf_model, normalize=True)

transformer_model = train_transformer(train, val)

evaluate_transformer_metrics("data", transformer_model)
transformer_errors = analyze_transformer_errors(OUT_DIR, transformer_model, normalize=False)
transformer_errors_normalized = analyze_transformer_errors(OUT_DIR, transformer_model, normalize=True)

# LABEL_LIST = ['I-SENTIMENT', 'O', 'I-ASPECT', 'B-SENTIMENT', 'B-ASPECT']
# ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
# tokens_test = ast.literal_eval(train["tokens"][1])
# labels_test = ast.literal_eval(train["seq_label"][1])
# labels_test = [ID2LABEL[l] for l in labels_test]

# demonstrate_alignment(tokens_test, labels_test)