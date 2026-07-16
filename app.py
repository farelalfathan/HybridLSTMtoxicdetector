"""
ClarityNLP - Toxic Comment Detector
Dashboard Streamlit untuk model Hybrid LSTM-GRU (klasifikasi biner toxic/non-toxic).

Cara pakai:
1. Taruh isi folder 'artifacts_for_streamlit' hasil export notebook (Cell 14) ke
   folder 'artifacts/' di sini (sejajar dengan file app.py ini).
   Isinya harus ada: model.keras, tokenizer.json, config.json, threshold.json, metrics.json
2. Jalankan lokal:  streamlit run app.py
3. Deploy ke Streamlit Cloud: push repo ini ke GitHub, lalu connect di
   https://share.streamlit.io -- pastikan folder artifacts/ ikut di-commit.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# =========================================
# KONFIGURASI HALAMAN
# =========================================
st.set_page_config(
    page_title="ClarityNLP - Toxic Comment Detector",
    page_icon="🛡️",
    layout="wide",
)

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

# Label baku dipakai konsisten di seluruh Penulisan Ilmiah (BAB II & BAB III):
# 0 = Non-Toxic, 1 = Toxic. "Hate speech" TIDAK dipakai sebagai label umum di
# sini karena itu subkategori spesifik (lihat Davidson et al., 2017 di BAB II),
# bukan sinonim dari toxic secara umum.
LABEL_MAPPING = {0: "Non-Toxic", 1: "Toxic"}


# =========================================
# LOAD ARTIFACTS (di-cache supaya tidak reload tiap interaksi)
# =========================================
@st.cache_resource(show_spinner="Memuat model Hybrid LSTM-GRU...")
def load_artifacts():
    required = ["model.keras", "tokenizer.json", "config.json", "threshold.json", "metrics.json"]
    missing = [f for f in required if not (ARTIFACT_DIR / f).exists()]
    if missing:
        st.error(
            "File artifact berikut tidak ditemukan di folder `artifacts/`: "
            + ", ".join(missing)
            + ".\n\nJalankan notebook training sampai Cell 14 (export untuk Streamlit), "
            "lalu ekstrak hasil zip-nya ke folder `artifacts/` di repo ini."
        )
        st.stop()

    model = tf.keras.models.load_model(ARTIFACT_DIR / "model.keras")

    with open(ARTIFACT_DIR / "tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    with open(ARTIFACT_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(ARTIFACT_DIR / "threshold.json", "r", encoding="utf-8") as f:
        threshold = json.load(f)["threshold"]

    with open(ARTIFACT_DIR / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return model, tokenizer, config, float(threshold), metrics


model, tokenizer, config, threshold, metrics = load_artifacts()
max_len = config.get("max_len", 180)


# =========================================
# CLEAN TEXT -- HARUS SAMA PERSIS DENGAN NOTEBOOK TRAINING (Cell 4)
# =========================================
def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"&amp;", " and ", text)

    # Apostrof dihapus TANPA menyisakan spasi (don't -> dont, bukan don t)
    text = text.replace("'", "").replace(chr(8217), "").replace('"', "")

    # Hanya sisakan huruf, angka, dan tanda baca dasar
    text = re.sub(r"[^a-z0-9!?.,\s]", " ", text)

    # Huruf berulang 3x+ direduksi jadi 2 huruf (sooooo -> soo), bukan 1,
    # supaya kata wajar berhuruf ganda (book, class) tidak ikut rusak.
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_comment(text: str):
    if not text or not text.strip():
        return None

    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    score = float(model.predict(padded, verbose=0)[0][0])
    label_code = 1 if score >= threshold else 0

    return {
        "comment": text,
        "clean": cleaned,
        "score": score,
        "threshold": threshold,
        "label_code": label_code,
        "label": LABEL_MAPPING[label_code],
    }


# =========================================
# SIDEBAR NAVIGASI
# =========================================
st.sidebar.title("🛡️ ClarityNLP")
st.sidebar.caption("Deteksi Komentar Toxic berbasis Hybrid LSTM-GRU")
page = st.sidebar.radio("Menu", ["📊 Dashboard", "🔍 Deteksi Komentar"])

st.sidebar.divider()
st.sidebar.markdown("**Info Model**")
st.sidebar.write(f"Threshold: `{threshold:.4f}`")
st.sidebar.write(f"Metrik seleksi: `{metrics.get('decision_metric', '-')}`")


# =========================================
# HALAMAN 1: DASHBOARD
# =========================================
if page == "📊 Dashboard":
    st.title("Dashboard Evaluasi Model")
    st.caption("Semua angka di halaman ini diambil langsung dari `metrics.json` hasil evaluasi pada data uji (test set) -- bukan angka contoh.")

    ds = metrics["dataset"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", f"{ds['total']:,}")
    col2.metric("Data Latih", f"{ds['train']:,}")
    col3.metric("Data Validasi", f"{ds['val']:,}")
    col4.metric("Data Uji", f"{ds['test']:,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Distribusi Label Dataset")
        fig, ax = plt.subplots(figsize=(5, 4))
        values = [ds["non_toxic_count"], ds["toxic_count"]]
        labels = [f"Non-Toxic\n({ds['non_toxic_ratio']:.1%})", f"Toxic\n({ds['toxic_ratio']:.1%})"]
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
               colors=["#2563eb", "#f97316"], textprops={"fontsize": 10, "fontweight": "bold"})
        ax.axis("equal")
        st.pyplot(fig)
        st.caption("Dataset bersifat tidak seimbang (imbalanced) -- lihat BAB II 2.6.4 Distribusi Data.")

    with right:
        st.subheader("Performa Model (Data Uji)")
        fig, ax = plt.subplots(figsize=(5, 4))
        metric_names = ["Accuracy", "Precision\n(toxic)", "Recall\n(toxic)", "F1-Score\n(toxic)", "ROC AUC"]
        metric_values = [
            metrics["accuracy"],
            metrics["toxic"]["precision"],
            metrics["toxic"]["recall"],
            metrics["toxic"]["f1_score"],
            metrics["roc_auc"],
        ]
        bars = ax.bar(metric_names, metric_values, color=["#2563eb", "#16a34a", "#f97316", "#9333ea", "#e11d48"])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, v in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)
        st.pyplot(fig)

    st.divider()

    st.subheader("Confusion Matrix (Data Uji)")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        [[cm["true_negative"], cm["false_positive"]], [cm["false_negative"], cm["true_positive"]]],
        index=["Aktual Non-Toxic", "Aktual Toxic"],
        columns=["Prediksi Non-Toxic", "Prediksi Toxic"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Classification Report Lengkap")
    report_df = pd.DataFrame({
        "Precision": [metrics["non_toxic"]["precision"], metrics["toxic"]["precision"],
                      metrics["macro_avg"]["precision"], metrics["weighted_avg"]["precision"]],
        "Recall": [metrics["non_toxic"]["recall"], metrics["toxic"]["recall"],
                   metrics["macro_avg"]["recall"], metrics["weighted_avg"]["recall"]],
        "F1-Score": [metrics["non_toxic"]["f1_score"], metrics["toxic"]["f1_score"],
                     metrics["macro_avg"]["f1_score"], metrics["weighted_avg"]["f1_score"]],
        "Support": [metrics["non_toxic"]["support"], metrics["toxic"]["support"], "-", "-"],
    }, index=["Non-Toxic (0)", "Toxic (1)", "Macro avg", "Weighted avg"])
    st.dataframe(report_df.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"}),
                 use_container_width=True)


# =========================================
# HALAMAN 2: DETEKSI KOMENTAR
# =========================================
else:
    st.title("Deteksi Komentar Toxic")
    st.caption("Model: Hybrid LSTM-GRU (paralel) | Threshold klasifikasi: " + f"{threshold:.4f}")

    col_input, col_result = st.columns([6, 5])

    with col_input:
        st.subheader("Input Komentar")
        text_input = st.text_area("Masukkan komentar", height=180,
                                   placeholder="Tulis komentar yang ingin dianalisis...")
        detect = st.button("Deteksi Komentar", type="primary", use_container_width=True)

        st.markdown("**Contoh komentar:**")
        examples = [
            "Have a nice day",
            "I appreciate your opinion",
            "Nobody likes you",
            "You are so stupid",
            "You are a useless idiot",
            "This has to be the worst take ever",
        ]
        example_cols = st.columns(2)
        selected_example = None
        for i, ex in enumerate(examples):
            if example_cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
                selected_example = ex

    with col_result:
        st.subheader("Hasil Deteksi")
        final_text = selected_example if selected_example else (text_input if detect else None)

        if final_text:
            result = predict_comment(final_text)
            if result is None:
                st.warning("Masukkan komentar terlebih dahulu.")
            else:
                if result["label_code"] == 1:
                    st.error(f"🔴 **Toxic** (skor: {result['score']:.4f})")
                    st.write("Komentar terindikasi mengandung unsur toxic dan perlu diperiksa/dimoderasi.")
                else:
                    st.success(f"🟢 **Non-Toxic** (skor: {result['score']:.4f})")
                    st.write("Komentar tidak terindikasi sebagai toxic berdasarkan threshold model.")

                st.divider()
                st.write("**Detail:**")
                detail_df = pd.DataFrame({
                    "Item": ["Skor Probabilitas", "Threshold", "Teks Setelah Cleaning"],
                    "Nilai": [f"{result['score']:.4f}", f"{result['threshold']:.4f}", result["clean"]],
                })
                st.dataframe(detail_df, use_container_width=True, hide_index=True)
        else:
            st.info("Masukkan komentar lalu klik **Deteksi Komentar**, atau pilih salah satu contoh di kiri.")

st.divider()
st.caption("Penulisan Ilmiah Klasifikasi Konten Toksik menggunakan Hybrid LSTM-GRU")
