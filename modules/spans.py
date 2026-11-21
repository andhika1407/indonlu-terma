I_SENT = "I-SENTIMENT"
B_SENT = "B-SENTIMENT"
I_ASPECT = "I-ASPECT"
B_ASPECT = "B-ASPECT"
O = "O"

TRANSITIONS = {
    I_SENT:   ({I_SENT, B_SENT},     B_SENT),
    I_ASPECT: ({I_ASPECT, B_ASPECT}, B_ASPECT),
}

def normalize_bio(labels):        
    prev = O
    for i, lbl in enumerate(labels):
        if lbl not in TRANSITIONS:
            prev = lbl
            continue
        allowed_prev, b_label = TRANSITIONS[lbl]
        if prev not in allowed_prev:
            labels[i] = b_label
            lbl = b_label
        prev = lbl
    return labels


def extract_labeled_spans(labels, normalize=True):
    """
    Extracts spans as a set of tuples: (entity_type, start_idx, end_idx_inclusive).
    Applies normalize_bio first to guarantee clean span extraction.
    """
    clean_labels = labels
    if normalize:
        clean_labels = normalize_bio(labels)
    
    spans = set()
    
    current_entity = None
    start_idx = None

    for i, lbl in enumerate(clean_labels):
        if lbl.startswith("B-"):
            if current_entity is not None:
                spans.add((current_entity, start_idx, i - 1))
            current_entity = lbl.split("-")[1]
            start_idx = i
        elif lbl.startswith("I-"):
            entity = lbl.split("-")[1]
            if current_entity == entity:
                pass  # Continuation
            else:
                # Should not happen after normalization, but safe fallback
                # I- ilegal tanpa B- dari entitas yang sama
                    if normalize:
                        # Perlakukan sebagai B- baru
                        if current_entity is not None:
                            spans.add((current_entity, start_idx, i - 1))
                        current_entity = entity
                        start_idx = i
                    else:
                        # Strict Raw, tutup span sebelumnya, ignore tag liar
                        if current_entity is not None:
                            spans.add((current_entity, start_idx, i - 1))
                        current_entity = None
                        start_idx = None
                        
        else:  # 'O'
            if current_entity is not None:
                spans.add((current_entity, start_idx, i - 1))
                current_entity = None
                start_idx = None

    if current_entity is not None:
        spans.add((current_entity, start_idx, len(clean_labels) - 1))

    return spans