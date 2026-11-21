def validate_annotation(row):
    """
    Validates the tokens annotations automatically:
    - checks for illegal transitions (e.g. O -> I-X, B-ASPECT -> I-SENTIMENT)
    - repair the transition based on previous token
    """
    labels  = row['seq_label']
    
    TRANSITIONS = {
        0: ([0, 3], 3),   # I-Sentiment -> sebelumnya harus I-Sentiment atau B-Sentiment, kalau bukan default jadi B-Sentiment
        2: ([2, 4], 4),   # I-Aspect    -> sebelumnya harus I-Aspect atau B-Aspect, kalau bukan default jadi B-Aspect
    }

    prev_label = 1  # O

    for lbl in labels:
        if lbl not in TRANSITIONS:      # skip kalau bukan I-Sentiment atau I-Aspect (karena di dictionary cuman ada index 0 dan 2)
            prev_label = lbl
            continue

        allowed_prev, b_label = TRANSITIONS[lbl]
        if prev_label not in allowed_prev:
            lbl = b_label               # Merubah dari I- jadi B- 

        prev_label = lbl

    return labels

def validate_annotation_manual(df, show_context=True):
    """
    Manually validates token annotations:
    - Checks for illegal transitions (e.g. O -> I-X, B-ASPECT -> I-SENTIMENT)
    - Shows the sentence and the anomalous token to the user
    - User decides whether it should be B- (default repair) or something else
    """

    df_copy = df.copy()  # don't mutate the caller's frame
    df_copy['seq_label_fixed'] = df_copy['seq_label']

    TRANSITIONS = {
        0: ([0, 3], 3),   # I-Sentiment: must follow I-Sentiment or B-Sentiment
        2: ([2, 4], 4),   # I-Aspect:    must follow I-Aspect or B-Aspect
    }

    LABEL_NAMES = {
        0: "I-SENTIMENT",
        1: "O",
        2: "I-ASPECT",
        3: "B-SENTIMENT",
        4: "B-ASPECT",
    }

    def find_first_anomaly(labels):
        """Return (index, lbl, prev_lbl, default_b) for first illegal
        transition, or None if the sequence is clean."""
        prev_label = 1  # O
        for i, lbl in enumerate(labels):
            if lbl in TRANSITIONS:
                allowed_prev, b_label = TRANSITIONS[lbl]
                if prev_label not in allowed_prev:
                    return (i, lbl, prev_label, b_label)
            prev_label = lbl
        return None
    
    for idx, row in df_copy.iterrows():
        tokens = row['tokens']
        labels = list(row['seq_label'])

        anomalies_fixed = 0
        while True:
            hit = find_first_anomaly(labels)
            if hit is None:
                break

            i, lbl, prev_lbl, default_b = hit
            anomalies_fixed += 1

            print("=" * 70)
            print(f"[row id={row['id']}]")
            print(f"Sentence: {' '.join(tokens)}")
            if show_context:
                lo, hi = max(0, i - 3), min(len(tokens), i + 4)
                for j in range(lo, hi):
                    marker = "  <<< ANOMALY" if j in (i, i - 1) else ""
                    print(f"  [{j}] {tokens[j]:20s} -> {LABEL_NAMES[labels[j]]}{marker}")
                    
            print(f"\nIllegal transition: [{i-1}] {LABEL_NAMES[prev_lbl]} -> [{i}] {LABEL_NAMES[lbl]}")
            print(f"Fix both tokens (press Enter to keep current label).")

            for target in (i - 1, i):
                cur = LABEL_NAMES[labels[target]]
                while True:
                    raw = input(
                        f"\n[{target}] '{tokens[target]}' (current: {cur}) "
                        f"→ \nnew label (0=I-SENTIMENT \n1=O \n2=I-ASPECT \n3=B-SENTIMENT \n4=B-ASPECT, "
                        f"\nEnter=keep): "
                    ).strip()
                    if raw == "":
                        print(f"  → kept as {cur}")
                        break
                    try:
                        num = int(raw)
                    except ValueError:
                        print("  Please enter a number 0–4, or Enter to keep.")
                        continue
                    if num not in LABEL_NAMES:
                        print("  Invalid label id, try again.")
                        continue
                    labels[target] = num
                    print(f"  → set [{target}] to {LABEL_NAMES[num]}")
                    break

            # loop back: re-scan the whole sentence with the updated labels

        if anomalies_fixed:
            print("=" * 70)
        df_copy.at[idx, 'seq_label_fixed'] = labels

    return df_copy
