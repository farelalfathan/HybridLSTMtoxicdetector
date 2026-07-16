import json
import re
import os
import gdown
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# =========================================
# KONFIGURASI
# =========================================
FILE_ID = "1z83SxY2HrK836-G9jM7G9w6f-fOHcMOR"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"

st.set_page_config(
    page_title="ClarityNLP - Toxic Comment Detector",
    page_icon="🛡️",
    layout="wide",
)

LABEL_MAPPING = {0: "Non-Toxic", 1: "Toxic"}

# =========================================
# DATASET (10 Per Section)
# =========================================
EXAMPLES = {
    "Non-Toxic": [
        "Thank you for your constructive feedback.",
        "I agree with the consensus reached here.",
        "Could you please provide a reliable source?",
        "The article needs more neutrality.",
        "Great job on cleaning up the references.",
        "Let's discuss the changes before editing.",
        "Thanks for clarifying, I understand now.",
        "That sounds like a fair compromise.",
        "I am trying to improve the quality.",
        "Your explanation makes perfect sense."
    ],
    "Toxic": [
        "You are an idiot and should be banned.",
        "Stop deleting my edits you stupid vandal.",
        "Go away, nobody wants you here.",
        "This article is absolute garbage.",
        "Shut up, you don't know anything.",
        "I hate you and your stupid contributions.",
        "You are a racist piece of trash.",
        "You are incompetent and have no idea what you are doing.",
        "You are a pathetic loser.",
        "Die in a fire, you miserable fool."
    ]
}

# =========================================
# LOAD ARTIFACTS
# =========================================
@st.cache_resource(show_spinner="Memuat model Hybrid LSTM-GRU...")
def load_artifacts():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "model.keras"

    if not model_path.exists():
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        with st.spinner("📥 Mengunduh model AI..."):
            try:
                gdown.download(url=url, output=str(model_path), quiet=True)
            except Exception as e:
                st.error(f"Gagal mengunduh: {e}")
                st.stop()

    required = ["model.keras", "tokenizer.json", "config.json", "threshold.json", "metrics.json"]
    missing = [f for f in required if not (ARTIFACT_DIR / f).exists()]
    if missing:
        st.error(f"Artifact hilang: {', '.join(missing)}")
        st.stop()

    model = tf.keras.models.load_model(model_path)
    with open(ARTIFACT_DIR / "tokenizer.json", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())
    with open(ARTIFACT_DIR / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    with open(ARTIFACT_DIR / "threshold.json", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])
    with open(ARTIFACT_DIR / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    return model, tokenizer, config, threshold, metrics

model, tokenizer, config, threshold, metrics = load_artifacts()
max_len = config.get("max_len", 180)

# =========================================
# FUNGSI PENDUKUNG
# =========================================
def clean_text(text: str) -> str:
    if not text: return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"&amp;", " and ", text)
    text = text.replace("'", "").replace(chr(8217), "").replace('"', "")
    text = re.sub(r"[^a-z0-9!?.,\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_comment(text: str):
    if not text or not text.strip(): return None
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    score = float(model.predict(padded, verbose=0)[0][0])
    label_code = 1 if score >= threshold else 0
    return {
        "clean": cleaned,
        "score": score,
        "label_code": label_code,
        "label": LABEL_MAPPING[label_code],
    }

# =========================================
# SIDEBAR
# =========================================
st.sidebar.title("🛡️ ClarityNLP")
st.sidebar.caption("Deteksi Komentar Toxic")
page = st.sidebar.radio("Menu", ["📊 Dashboard", "🔍 Deteksi Komentar"])

st.sidebar.divider()
st.sidebar.markdown("**Info Model**")
st.sidebar.write(f"Threshold: `{threshold:.4f}`")
st.sidebar.write(f"Metrik: `{metrics.get('decision_metric', '-')}`")

# =========================================
# HALAMAN 1: DASHBOARD
# =========================================
if page == "📊 Dashboard":
    st.title("Dashboard Evaluasi Model")
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
        fig, ax = plt.subplots(figsize=(4, 4))
        values = [ds["non_toxic_count"], ds["toxic_count"]]
        labels = [f"Non-Toxic\n({ds['non_toxic_ratio']:.1%})", f"Toxic\n({ds['toxic_ratio']:.1%})"]
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
               colors=["#2563eb", "#f97316"], textprops={"fontsize": 10, "fontweight": "bold"})
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with right:
        st.subheader("Performa Model (Data Uji)")
        fig, ax = plt.subplots(figsize=(6, 4))
        metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"]
        metric_values = [metrics["accuracy"], metrics["toxic"]["precision"], metrics["toxic"]["recall"], metrics["toxic"]["f1_score"], metrics["roc_auc"]]
        bars = ax.bar(metric_names, metric_values, color=["#2563eb", "#16a34a", "#f97316", "#9333ea", "#e11d48"])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, v in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.divider()
    st.subheader("Confusion Matrix")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        [[cm["true_negative"], cm["false_positive"]], [cm["false_negative"], cm["true_positive"]]],
        index=["Aktual Non-Toxic", "Aktual Toxic"],
        columns=["Prediksi Non-Toxic", "Prediksi Toxic"],
    )
    st.dataframe(cm_df, use_container_width=True)

# =========================================
# HALAMAN 2: DETEKSI KOMENTAR
# =========================================
else:
    st.title("Deteksi Komentar Toxic")
    if "test_text" not in st.session_state: st.session_state.test_text = ""

    col_input, col_result = st.columns([6, 5])
    
    with col_input:
        text_input = st.text_area("Masukkan komentar", height=180, placeholder="Tulis komentar...", value=st.session_state.test_text)
        detect = st.button("Deteksi Komentar", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.write("💡 **Quick Test:**")
        sample_cols = st.columns(2)
        if sample_cols[0].button("Contoh Non-Toxic"):
            st.session_state.test_text = random.choice(EXAMPLES["Non-Toxic"])
            st.rerun()
        if sample_cols[1].button("Contoh Toxic"):
            st.session_state.test_text = random.choice(EXAMPLES["Toxic"])
            st.rerun()

    with col_result:
        st.subheader("Hasil Deteksi")
        if detect and text_input:
            with st.spinner("AI sedang menganalisis..."):
                time.sleep(0.8) 
                result = predict_comment(text_input)
            
            if result:
                if result["label_code"] == 1:
                    st.error(f"🔴 **Toxic** (skor: {result['score']:.4f})")
                    st.toast("⚠️ Terdeteksi konten Toxic!", icon="🚨")
                else:
                    st.success(f"🟢 **Non-Toxic** (skor: {result['score']:.4f})")
                    st.balloons()
                    st.toast("✅ Komentar aman dan santun", icon="✨")
                
                st.markdown("---")
                st.write(f"**Teks Clean:** `{result['clean']}`")
                st.progress(result['score'])
                st.write("Skor mendekati 1 artinya semakin tinggi probabilitas toxic.")
        
        elif detect and not text_input:
            st.warning("Silakan masukkan teks terlebih dahulu.")
