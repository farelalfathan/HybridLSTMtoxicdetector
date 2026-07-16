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

st.set_page_config(page_title="ClarityNLP - Toxic Comment Detector", page_icon="🛡️", layout="wide")
LABEL_MAPPING = {0: "Non-Toxic", 1: "Toxic"}

# =========================================
# DATASET 200 SAMPLE (100 Toxic, 100 Non-Toxic)
# =========================================
EXAMPLES = {
    "Non-Toxic": [
        "Thanks for your feedback on my edits.", "I agree with the consensus reached on this talk page.",
        "Could you please provide a source for this claim?", "I think the article needs more neutrality.",
        "Great work on cleaning up the references.", "I am just trying to improve the quality of this article.",
        "Let's discuss the changes before editing the main page.", "Thanks for clarifying, I understand now.",
        "The edit is constructive and follows Wikipedia guidelines.", "That sounds like a fair compromise.",
        "This is a well-researched argument.", "I appreciate your patience in explaining this.",
        "This article is very informative and clear.", "Can we add more details about this topic?",
        "I support this modification.", "Your points are very valid.", "This looks much better now.",
        "Thank you for your hard work on this page.", "Let's keep it civil and focus on facts.",
        "I've checked the guidelines, it seems correct.", "Everything seems to be in order here.",
        "Interesting perspective, I hadn't thought of that.", "Good catch on that typo.",
        "I agree with the proposed changes.", "Seems like a reasonable edit.", "Thanks for the help.",
        "The tone of this article is perfect.", "That’s a great suggestion.", "I appreciate the collaboration.",
        "Well written and very accurate.", "This is a significant improvement.", "Good job everyone.",
        "Let's stick to the evidence.", "I’m happy with this outcome.", "This adds value to the topic.",
        "Looks like we reached an agreement.", "Excellent point, I agree.", "Thanks for fixing that error.",
        "This is exactly what was needed.", "Neutral point of view is maintained here.",
        "Very concise and easy to understand.", "This has improved the clarity.", "I fully support this change.",
        "Constructive criticism is always good.", "This helps the community grow.", "Well argued point.",
        "Thanks for the heads up.", "Let's keep up the momentum.", "This is a stable version now.",
        "Good effort on this edit.", "This clarification is helpful.", "Great contribution to the discussion.",
        "I understand the point you are making.", "This makes a lot of sense.", "Thank you for the guidance.",
        "This is a balanced perspective.", "I agree with the cited source.", "That is an interesting link.",
        "Glad to see more activity here.", "This is a positive step forward.", "Thanks for updating the info.",
        "I appreciate your honesty.", "Your summary is quite accurate.", "This helps clarify the situation.",
        "We should follow the standard procedure.", "That is a valid concern.", "I see why you did that.",
        "Great job keeping the discussion civil.", "This is a good summary of the events.",
        "Thanks for maintaining neutrality.", "I’ve reviewed the edit, it looks fine.",
        "That is an excellent counter-point.", "Let's look at the evidence together.",
        "This is a very detailed explanation.", "I agree, let's proceed.", "Thanks for the heads up.",
        "Your input is appreciated.", "This is consistent with the policy.", "Looks good to me.",
        "I support the new consensus.", "Well reasoned response.", "Thanks for the clarification.",
        "This is an interesting historical fact.", "Keep up the great work.", "That is a fair assessment.",
        "I agree with the logic provided.", "Thank you for the explanation.", "This is accurate.",
        "I like how you explained that.", "This is very well structured.", "Thanks for your time.",
        "I support this decision.", "Good discussion, everyone.", "That works for me."
    ],
    "Toxic": [
        "You are an idiot and should be banned from Wikipedia.", "Stop deleting my edits you stupid vandal.",
        "Go kill yourself, nobody wants you here.", "This article is total garbage, just like the author.",
        "Shut up, you don't know anything about this topic.", "I hate you and your stupid contributions.",
        "You're a racist piece of trash.", "Get lost, nobody cares about your opinion.",
        "Your mother is a whore.", "Die in a fire, you loser.", "You are pathetic.",
        "Everyone hates your guts here.", "This is the worst article I have ever read.",
        "You are so stupid it hurts.", "Don't ever talk to me again you creep.",
        "What a waste of space you are.", "Go suck a lemon you retard.", "You are brainless.",
        "Nobody likes you and you know it.", "Shut your mouth, you know nothing.",
        "Stop being such a douchebag.", "You are a disgrace to this community.",
        "I can't believe how dumb you are.", "You deserve to be banned forever.",
        "You're a complete loser with no life.", "This entire project is a disaster thanks to you.",
        "Go jump off a bridge.", "You are annoying as hell.", "Leave now you pathetic clown.",
        "Your opinion means absolutely nothing to me.", "Stop acting like a know-it-all.",
        "You are an absolute waste of time.", "Why are you even allowed to post here?",
        "Everything you say is a lie.", "You are truly disgusting.", "Get a life you pathetic weirdo.",
        "This is pure filth.", "You are clearly too stupid to understand.",
        "Your existence is an annoyance to me.", "Shut up before you make a bigger fool of yourself.",
        "You are incompetent and useless.", "I loathe everything you stand for.",
        "Just give up already.", "You are a total jerk.", "Who asked for your input, idiot?",
        "You are a cancer to this site.", "Stop bothering everyone with your stupidity.",
        "You are absolutely repulsive.", "Get out of here before I get angry.", "You are a joke.",
        "Your brain must be the size of a pea.", "You are scum of the earth.", "You are a waste of oxygen.",
        "Nobody here respects you.", "Shut the fuck up.", "You are ugly and stupid.",
        "Go crawl back into the hole you came from.", "Stop spreading your lies.", "You are so fake.",
        "You are manipulative and evil.", "I wish you would just disappear.", "You are trash.",
        "Your work is worthless.", "Stop trying to be smart.", "You are failing at life.",
        "You make me sick.", "You are a parasite.", "Nobody will ever love you.",
        "You are a failure.", "You are delusional.", "Stop being so arrogant.",
        "You are beneath me.", "You are a laughing stock.", "You are a stain on humanity.",
        "You are pathetic and weak.", "Shut your trap.", "Stop wasting our time.",
        "You are a loser.", "You are a fraud.", "You are an embarrassment.",
        "Go away.", "You are toxic.", "You are the worst person I have ever met.",
        "You are ignorant.", "You are so loud and annoying.", "You are a nuisance.",
        "You are a parasite to this platform.", "I hate reading your comments.", "You are so predictable."
    ]
}

# =========================================
# LOAD ARTIFACTS (Keep existing function)
# =========================================
@st.cache_resource(show_spinner="Memuat model Hybrid LSTM-GRU...")
def load_artifacts():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "model.keras"

    if not model_path.exists():
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url=url, output=str(model_path), quiet=True)

    required = ["model.keras", "tokenizer.json", "config.json", "threshold.json", "metrics.json"]
    if any(not (ARTIFACT_DIR / f).exists() for f in required):
        st.error("Artifact hilang!")
        st.stop()

    model = tf.keras.models.load_model(model_path)
    with open(ARTIFACT_DIR / "tokenizer.json", encoding="utf-8") as f: tokenizer = tokenizer_from_json(f.read())
    with open(ARTIFACT_DIR / "config.json", encoding="utf-8") as f: config = json.load(f)
    with open(ARTIFACT_DIR / "threshold.json", encoding="utf-8") as f: threshold = float(json.load(f)["threshold"])
    with open(ARTIFACT_DIR / "metrics.json", encoding="utf-8") as f: metrics = json.load(f)
    return model, tokenizer, config, threshold, metrics

model, tokenizer, config, threshold, metrics = load_artifacts()
max_len = config.get("max_len", 180)

# =========================================
# FUNGSI PREDIKSI
# =========================================
def clean_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9!?.,\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()

def predict_comment(text: str):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    score = float(model.predict(padded, verbose=0)[0][0])
    return {"clean": cleaned, "score": score, "label_code": 1 if score >= threshold else 0}

# =========================================
# UI
# =========================================
st.sidebar.title("🛡️ ClarityNLP")
page = st.sidebar.radio("Menu", ["📊 Dashboard", "🔍 Deteksi Komentar"])

if page == "📊 Dashboard":
    st.title("Dashboard Evaluasi")
    st.write("Dashboard aktif.")

else:
    st.title("Deteksi Komentar Toxic")
    if "test_text" not in st.session_state: st.session_state.test_text = ""
    col_input, col_result = st.columns([6, 5])
    
    with col_input:
        text_input = st.text_area("Masukkan komentar", height=180, value=st.session_state.test_text)
        detect = st.button("Deteksi Komentar", type="primary", use_container_width=True)
        st.markdown("---")
        st.write("💡 **Quick Test (200 Dataset):**")
        
        c1, c2 = st.columns(2)
        if c1.button("Random Non-Toxic"):
            st.session_state.test_text = random.choice(EXAMPLES["Non-Toxic"])
            st.rerun()
        if c2.button("Random Toxic"):
            st.session_state.test_text = random.choice(EXAMPLES["Toxic"])
            st.rerun()

    with col_result:
        if detect and text_input:
            with st.spinner("AI sedang berpikir..."):
                time.sleep(0.5)
                res = predict_comment(text_input)
            if res["label_code"] == 1:
                st.error(f"🔴 **Toxic** (skor: {res['score']:.4f})")
                st.toast("⚠️ Terdeteksi konten Toxic!", icon="🚨")
            else:
                st.success(f"🟢 **Non-Toxic** (skor: {res['score']:.4f})")
                st.balloons()
