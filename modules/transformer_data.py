from transformers import AutoTokenizer
from .config import MODEL_NAME, MAX_LEN, LABEL2ID, ID2LABEL

def create_tokenizer(model_name = MODEL_NAME):
    """
    Loads pretrained Fast Tokenizer for IndoBERT.
    """
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(tokens, labels, tokenizer, label2id = LABEL2ID, max_length = MAX_LEN):
    """
    Subword tokenization and label alignment policy (NLP-T5-02):
    - First subword of a token receives the original label ID.
    - Subsequent subwords of the same token receive -100 (ignored in loss computation).
    - Special tokens ([CLS], [SEP], [PAD]) receive -100.
    """
    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_offsets_mapping=False
    )

    word_ids = tokenized_inputs.word_ids()
    aligned_labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            # Special token ([CLS], [SEP])
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            # First subword of this word: assign true label ID
            label_str = labels[word_idx]
            aligned_labels.append(label2id.get(label_str, 0))
        else:
            # Continuation subword: assign -100 to avoid double counting in loss
            aligned_labels.append(-100)
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

def align_predictions_back_to_words(word_ids, subword_preds, id2label):
    """
    Maps subword-level model predictions back to the original word/token level (NLP-T5-03):
    Extracts the prediction corresponding to the first subword of each word.
    """
    word_preds = []
    previous_word_idx = None

    for word_idx, pred_id in zip(word_ids, subword_preds):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            word_preds.append(id2label.get(pred_id, "O"))
            previous_word_idx = word_idx

    return word_preds

def demonstrate_alignment(tokens, labels, tokenizer=None):
    """
    Demonstrates subword alignment on tokens and their labels.
    
    Args:
        tokens: List of word tokens, e.g. ["i", "am", "you"]
        labels: List of label IDs (parallel to tokens), e.g. [0, 1, 2, 3]
        tokenizer: Optional pre-built tokenizer. If None, will try to create one.
    """
    print("Testing subword alignment demonstration...")
    
    if tokenizer is None:
        try:
            tokenizer = create_tokenizer()
        except Exception as e:
            print(f"Online tokenizer download failed ({e}), creating offline demonstration.")
            return

    encoded = tokenize_and_align_labels(tokens, labels, tokenizer)
    subwords = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    aligned = encoded["labels"]

    print("\n--- Subword Alignment Demonstration ---")
    print(f"{'Subword':<20} | {'Label ID':<10} | {'Interpretation'}")
    print("-" * 60)
    for sw, lbl in zip(subwords, aligned):
        if lbl == -100:
            interp = "Ignored (-100) — continuation subword / special token"
        else:
            label_name = ID2LABEL.get(lbl, f"UNKNOWN({lbl})")
            interp = f"{label_name}"
        print(f"{sw:<20} | {lbl:<10} | {interp}")

    print("\n--- Word → Subword Mapping ---")
    for tok, lbl in zip(tokens, labels):
        print(f"  {tok!r:<20} → {ID2LABEL.get(lbl, lbl)}")
