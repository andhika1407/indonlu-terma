import os
import sys
import torch
import joblib
import streamlit as st

# Pastikan working directory = root project agar modul bisa diimport
sys.path.insert(0, os.path.dirname(__file__))

from modules.crf_features import sentence2features
from modules.spans import extract_labeled_spans
from modules.config import ID2LABEL, LABEL_LIST, LABEL2ID
from modules.transformer_data import create_tokenizer, align_predictions_back_to_words

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "EXPERIMENTS")

# ---------------------------------------------------------------------------
# Warna label
# ---------------------------------------------------------------------------
LABEL_COLORS = {
    "ASPECT":    {"bg": "#DBEAFE", "border": "#3B82F6", "text": "#1D4ED8"},   # biru
    "SENTIMENT": {"bg": "#D1FAE5", "border": "#10B981", "text": "#065F46"},   # hijau
}

# ---------------------------------------------------------------------------
# Cache model agar tidak di-reload setiap interaksi
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model CRF…")
def load_crf():
    path = os.path.join(EXPERIMENTS_DIR, "crf_best_model.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner="Memuat model Transformer…")
def load_transformer():
    from transformers import AutoModelForTokenClassification
    model_dir = os.path.join(EXPERIMENTS_DIR, "transformer_best_model")
    if not os.path.isdir(model_dir):
        return None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = create_tokenizer(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Tokenisasi sederhana (split by whitespace, bersihkan tanda baca di tepi)
# ---------------------------------------------------------------------------
def simple_tokenize(text: str):
    import re
    # Pisahkan tanda baca di awal/akhir kata sebagai token tersendiri
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return tokens


# ---------------------------------------------------------------------------
# Prediksi CRF
# ---------------------------------------------------------------------------
def predict_crf(tokens, model):
    X = sentence2features([tokens])
    preds = model.predict(X)
    return preds[0]   # list[str] label per token


# ---------------------------------------------------------------------------
# Prediksi Transformer
# ---------------------------------------------------------------------------
def predict_transformer(tokens, model, tokenizer):
    device = next(model.parameters()).device
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        return_tensors="pt",
    )
    word_ids = encoded.word_ids()
    input_ids      = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    subword_preds = outputs.logits.argmax(dim=-1)[0].cpu().tolist()
    return align_predictions_back_to_words(word_ids, subword_preds, ID2LABEL)


# ---------------------------------------------------------------------------
# Susun hasil ekstraksi dari labels → list span dict
# ---------------------------------------------------------------------------
def build_results(tokens, labels):
    spans = extract_labeled_spans(labels, normalize=True)
    aspects    = []
    sentiments = []
    for ent, s, e in sorted(spans, key=lambda x: x[1]):
        text = " ".join(tokens[s : e + 1])
        if ent == "ASPECT":
            aspects.append(text)
        elif ent == "SENTIMENT":
            sentiments.append(text)
    return aspects, sentiments


# ---------------------------------------------------------------------------
# Render highlighted sentence
# ---------------------------------------------------------------------------
def render_highlighted(tokens, labels):
    html_parts = []
    i = 0
    # Bangun dict index → (ent, is_start, is_end) dari spans
    span_map = {}
    spans = extract_labeled_spans(labels, normalize=True)
    for ent, s, e in spans:
        for idx in range(s, e + 1):
            span_map[idx] = (ent, idx == s, idx == e)

    for i, (tok, lbl) in enumerate(zip(tokens, labels)):
        if i in span_map:
            ent, is_start, is_end = span_map[i]
            c = LABEL_COLORS[ent]
            radius_left  = "8px" if is_start else "0"
            radius_right = "8px" if is_end   else "0"
            style = (
                f"background:{c['bg']}; color:{c['text']}; border:1.5px solid {c['border']}; "
                f"border-radius:{radius_left} {radius_right} {radius_right} {radius_left}; "
                f"padding:2px 5px; margin:1px 0; display:inline-block;"
            )
            html_parts.append(f'<span style="{style}">{tok}</span>')
        else:
            html_parts.append(f'<span style="padding:2px 3px; display:inline-block;">{tok}</span>')

    return " ".join(html_parts)


# ---------------------------------------------------------------------------
# Badge chip helper
# ---------------------------------------------------------------------------
def chip(text, ent_type):
    c = LABEL_COLORS[ent_type]
    return (
        f'<span style="background:{c["bg"]}; color:{c["text"]}; '
        f'border:1.5px solid {c["border"]}; border-radius:16px; '
        f'padding:4px 12px; margin:4px 4px; font-size:0.9rem; '
        f'display:inline-block;">'
        f'  {text}'
        f'</span>'
    )


# ===========================================================================
# STREAMLIT UI
# ===========================================================================

st.set_page_config(
    page_title="ABSA Extractor — IndoNLU TERMA",
    page_icon="🔍",
    layout="wide",
)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Pengaturan")
    model_choice = st.selectbox(
        "Pilih Model",
        ["CRF", "Transformer"],
        index=0,
        help="CRF: lebih cepat, berbasis fitur.\nTransformer: IndoBERT fine-tuned."
    )
    normalize = st.toggle("Terapkan normalize_bio", value=True,
                          help="Perbaiki urutan BIO yang tidak valid sebelum ekstraksi span.")
    st.divider()
    st.caption("**Label:**")
    for ent, c in LABEL_COLORS.items():
        st.markdown(
            f'<span style="background:{c["bg"]};color:{c["text"]};'
            f'border:1.5px solid {c["border"]};border-radius:8px;'
            f'padding:3px 10px;font-size:0.85rem;">{ent}</span>',
            unsafe_allow_html=True
        )
    st.divider()
    st.caption("Model tersimpan di `EXPERIMENTS/`")


# --- Header ---
st.title("Aspect & Sentiment Extractor")
st.caption("Ekstraksi aspek dan sentimen dari komentar berbahasa Indonesia · Dataset IndoNLU TERMA")

st.divider()

# --- Layout utama: input kiri, output kanan ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Input Komentar")
    user_input = st.text_area(
        label="Masukkan komentar di sini",
        placeholder="Contoh: Makanan di sini sangat enak tapi pelayanannya lambat.",
        height=150,
        label_visibility="collapsed",
    )

    extract_btn = st.button("Extract", type="primary", use_container_width=True)

with col_right:
    st.subheader("Hasil Ekstraksi")
    result_placeholder = st.empty()

# --- Proses saat tombol ditekan ---
if extract_btn:
    if not user_input.strip():
        with col_right:
            st.warning("Masukkan komentar terlebih dahulu.")
    else:
        tokens = simple_tokenize(user_input.strip())

        if model_choice == "CRF":
            crf_model = load_crf()
            if crf_model is None:
                with col_right:
                    st.error("Model CRF tidak ditemukan di `EXPERIMENTS/crf_best_model.joblib`.")
                st.stop()
            with st.spinner("Memproses dengan CRF…"):
                labels = predict_crf(tokens, crf_model)
        else:
            transformer_model, tokenizer = load_transformer()
            if transformer_model is None:
                with col_right:
                    st.error("Model Transformer tidak ditemukan di `EXPERIMENTS/transformer_best_model/`.")
                st.stop()
            with st.spinner("Memproses dengan Transformer…"):
                labels = predict_transformer(tokens, transformer_model, tokenizer)

        # Pastikan panjang labels sama dengan tokens (safety guard)
        if len(labels) != len(tokens):
            labels = (labels + ["O"] * len(tokens))[: len(tokens)]

        aspects, sentiments = build_results(tokens, labels)

        with col_right:
            # Kalimat ter-highlight
            st.markdown("**Visualisasi Token:**")
            highlighted_html = render_highlighted(tokens, labels)
            st.markdown(
                f'<div style="line-height:2.2; font-size:1rem; '
                f'padding:12px; border-radius:10px; '
                f'background:#F8FAFC; border:1px solid #E2E8F0;">'
                f'{highlighted_html}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            # Aspek
            st.markdown("**Aspek yang ditemukan:**")
            if aspects:
                chips_html = "".join(chip(a, "ASPECT") for a in aspects)
                st.markdown(f'<div>{chips_html}</div>', unsafe_allow_html=True)
            else:
                st.info("Tidak ada aspek terdeteksi.")

            st.markdown("")

            # Sentimen
            st.markdown("**Sentimen yang ditemukan:**")
            if sentiments:
                chips_html = "".join(chip(s, "SENTIMENT") for s in sentiments)
                st.markdown(f'<div>{chips_html}</div>', unsafe_allow_html=True)
            else:
                st.info("Tidak ada sentimen terdeteksi.")

            # Detail token-label (expandable)
            with st.expander("Detail label per token"):
                import pandas as pd
                df_detail = pd.DataFrame({"Token": tokens, "Label": labels})
                st.dataframe(df_detail, use_container_width=True, hide_index=True)

# --- Footer ---
st.divider()
st.caption(
    f"Model aktif: **{model_choice}** · "
    f"normalize\\_bio: **{'ON' if normalize else 'OFF'}** · "
    f"Device: **{'CUDA' if torch.cuda.is_available() else 'CPU'}**"
)
