# ============================================================
#  app.py  –  20 Newsgroups Text Classifier  |  Streamlit GUI
#  Usage:  streamlit run app.py
#  Requires: model.pkl, vectorizer.pkl, labels.pkl
#            (downloaded from Colab OR run train_and_save.py)
#  All 3 files must be in the SAME folder as app.py
# ============================================================

import os
import pickle
import numpy as np
import streamlit as st

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="News Category Classifier",
    page_icon="📰",
    layout="centered",
)

# ── CUSTOM CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #0d1117; color: #c9d1d9; }

    .main-header {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        border-top: 3px solid #58a6ff;
        border-radius: 10px;
        padding: 28px 32px;
        margin-bottom: 28px;
    }
    .main-header h1 {
        font-family: 'IBM Plex Mono', monospace;
        color: #e6edf3; font-size: 1.7rem; margin: 0; letter-spacing: -0.5px;
    }
    .main-header p { color: #8b949e; margin: 8px 0 0; font-size: 0.9rem; line-height: 1.6; }

    .stTextArea textarea {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.88rem !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
    }

    .stButton > button {
        background: #238636; color: #ffffff;
        border: 1px solid #2ea043;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600; font-size: 0.95rem;
        padding: 12px 0; border-radius: 8px; width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover { background: #2ea043; border-color: #3fb950; }

    .result-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        border-radius: 10px;
        padding: 24px 28px;
        margin-top: 24px;
    }
    .result-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem; color: #8b949e;
        letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px;
    }
    .result-category {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.55rem; font-weight: 600;
        color: #58a6ff; margin: 0 0 4px; word-break: break-word;
    }
    .result-group { font-size: 0.85rem; color: #8b949e; margin-bottom: 18px; }

    .cat-badge {
        display: inline-block;
        background: rgba(88,166,255,0.1);
        border: 1px solid rgba(88,166,255,0.3);
        color: #58a6ff;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem; padding: 3px 10px;
        border-radius: 20px; margin-bottom: 20px;
    }

    .top-preds { width: 100%; border-collapse: collapse; margin-top: 16px; }
    .top-preds th {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.66rem; letter-spacing: 1.5px;
        text-transform: uppercase; color: #8b949e;
        border-bottom: 1px solid #30363d;
        padding: 6px 0; text-align: left;
    }
    .top-preds td {
        font-size: 0.84rem; color: #c9d1d9;
        padding: 9px 0; border-bottom: 1px solid #21262d; vertical-align: middle;
    }
    .top-preds tr:last-child td { border-bottom: none; }
    .bar-bg { background: #21262d; border-radius: 4px; height: 8px; width: 100%; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 4px; }

    .disclaimer {
        background: #0d1117; border: 1px solid #21262d;
        border-radius: 6px; padding: 10px 14px;
        font-size: 0.78rem; color: #6e7681;
        margin-top: 18px; line-height: 1.6;
    }

    label { color: #8b949e !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ── CATEGORY GROUPS ──────────────────────────────────────────
CATEGORY_GROUPS = {
    "alt.atheism":             "Religion & Philosophy",
    "comp.graphics":           "Computers & Tech",
    "comp.os.ms-windows.misc": "Computers & Tech",
    "comp.sys.ibm.pc.hardware":"Computers & Tech",
    "comp.sys.mac.hardware":   "Computers & Tech",
    "comp.windows.x":          "Computers & Tech",
    "misc.forsale":            "Marketplace",
    "rec.autos":               "Recreation & Hobbies",
    "rec.motorcycles":         "Recreation & Hobbies",
    "rec.sport.baseball":      "Recreation & Hobbies",
    "rec.sport.hockey":        "Recreation & Hobbies",
    "sci.crypt":               "Science & Research",
    "sci.electronics":         "Science & Research",
    "sci.med":                 "Science & Research",
    "sci.space":               "Science & Research",
    "soc.religion.christian":  "Religion & Philosophy",
    "talk.politics.guns":      "Politics & Society",
    "talk.politics.mideast":   "Politics & Society",
    "talk.politics.misc":      "Politics & Society",
    "talk.religion.misc":      "Religion & Philosophy",
}

# ── EXAMPLE TEXTS ────────────────────────────────────────────
EXAMPLES = {
    "🚀 Space":        "NASA announced a new mission to Mars next year. The spacecraft will carry advanced instruments to study the Martian atmosphere and search for signs of ancient microbial life on the red planet.",
    "💻 Tech (Windows)":"Windows drivers for the new graphics card can be installed via device manager. The BIOS update is required first before the display adapter is recognized by the operating system.",
    "⚾ Baseball":      "The pitcher threw a perfect game last night, striking out 12 batters. The team batting average has improved significantly since the new coach joined the club this season.",
    "🔬 Medicine":      "Researchers found that the new drug significantly reduces inflammation in patients with rheumatoid arthritis. Clinical trials showed a 60 percent improvement in joint pain symptoms after 8 weeks.",
    "🏍️ Motorcycles":   "The new Harley Davidson touring bike has excellent suspension and a powerful 1800cc V-twin engine. Highway riding at 70mph feels very smooth and stable with the new chassis design.",
    "🔐 Cryptography":  "RSA encryption uses public and private key pairs based on prime factorization. The security of the algorithm depends on the computational difficulty of factoring very large numbers efficiently.",
}

# ── LOAD ARTIFACTS ───────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
LABELS_PATH     = os.path.join(BASE_DIR, "labels.pkl")

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH,       "rb") as f: model      = pickle.load(f)
    with open(VECTORIZER_PATH,  "rb") as f: vectorizer = pickle.load(f)
    with open(LABELS_PATH,      "rb") as f: labels     = pickle.load(f)
    return model, vectorizer, list(labels)

# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📰 News Category Classifier</h1>
  <p>Paste or type any text — the model predicts which of the
     <strong>20 newsgroup categories</strong> it belongs to.<br>
     Dataset: <em>20 Newsgroups (Scikit-learn)</em> &nbsp;·&nbsp;
     Models evaluated: <em>LinearSVC · SGDClassifier · MultinomialNB</em></p>
</div>
""", unsafe_allow_html=True)

# ── CHECK FILES ──────────────────────────────────────────────
missing = [p for p in [MODEL_PATH, VECTORIZER_PATH, LABELS_PATH] if not os.path.exists(p)]
if missing:
    st.error(
        f"⚠️ Missing file(s): `{'`, `'.join(os.path.basename(p) for p in missing)}`\n\n"
        "Place **model.pkl**, **vectorizer.pkl**, and **labels.pkl** "
        "(downloaded from Colab) in the **same folder** as `app.py`, then restart."
    )
    st.stop()

model, vectorizer, labels = load_artifacts()

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Options")
    top_n = st.slider("Top N predictions", min_value=3, max_value=10, value=5)
    st.markdown("---")
    st.caption(f"**Categories:** {len(labels)}")
    st.caption(f"**Model:** {type(model).__name__}")
    st.caption("**Vectorizer:** TF-IDF · unigrams + bigrams")
    st.markdown("---")
    st.markdown("**📋 Load an example:**")
    for ex_label, ex_text in EXAMPLES.items():
        if st.button(ex_label, key=f"ex_{ex_label}"):
            st.session_state["input_text"] = ex_text

# ── INPUT ────────────────────────────────────────────────────
st.markdown("### ✍️ Enter Text to Classify")

user_text = st.text_area(
    label="Paste or type your text below:",
    value=st.session_state.get("input_text", ""),
    height=210,
    placeholder="e.g.  The Hubble Space Telescope captured stunning new images of a distant galaxy...",
    key="text_input",
)

col1, col2 = st.columns([4, 1])
with col1:
    predict_btn = st.button("🔍  Classify Text", use_container_width=True)
with col2:
    if st.button("✕ Clear", use_container_width=True):
        st.session_state["input_text"] = ""
        st.rerun()

# ── PREDICTION ───────────────────────────────────────────────
if predict_btn:
    text = user_text.strip()

    if not text:
        st.warning("⚠️ Please enter some text before classifying.")
    elif len(text.split()) < 3:
        st.warning("⚠️ Please enter at least a few words for a meaningful prediction.")
    else:
        # Vectorize input → predict
        X_input    = vectorizer.transform([text])
        prediction = model.predict(X_input)[0]
        pred_label = labels[prediction]
        pred_group = CATEGORY_GROUPS.get(pred_label, "General")

        # Confidence scores via decision_function (softmax) or predict_proba
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_input)[0]
        elif hasattr(model, "decision_function"):
            raw    = model.decision_function(X_input)[0]
            raw    = raw - raw.max()          # numerical stability
            exp_r  = np.exp(raw)
            scores = exp_r / exp_r.sum()      # softmax → relative confidence
        else:
            scores = np.zeros(len(labels))
            scores[prediction] = 1.0

        top_idx    = np.argsort(scores)[::-1][:top_n]
        top_scores = scores[top_idx]
        top_labels = [labels[i] for i in top_idx]
        confidence = float(scores[prediction]) * 100

        # ── Build bar rows ────────────────────────────────────
        max_score = top_scores[0] if top_scores[0] > 0 else 1.0
        bars_html = ""
        for lbl, sc in zip(top_labels, top_scores):
            pct      = sc * 100
            bar_pct  = (sc / max_score) * 100          # relative width
            is_best  = (lbl == pred_label)
            bar_color= "#58a6ff" if is_best else "#1f6feb"
            bold     = "font-weight:600; color:#e6edf3;" if is_best else ""
            bars_html += f"""
            <tr>
              <td style="{bold} width:42%; padding-right:10px;">{lbl}</td>
              <td style="width:44%; padding-right:14px;">
                <div class="bar-bg">
                  <div class="bar-fill" style="width:{bar_pct:.1f}%; background:{bar_color};"></div>
                </div>
              </td>
              <td style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
                         color:#8b949e; text-align:right; white-space:nowrap;">
                {pct:.1f}%
              </td>
            </tr>"""

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Category</div>
            <div class="result-category">{pred_label}</div>
            <div class="result-group">📁 {pred_group}</div>
            <div class="cat-badge">Confidence&nbsp; {confidence:.1f}%</div>

            <div class="result-label">Top {top_n} Predictions</div>
            <table class="top-preds">
              <tr>
                <th>Category</th>
                <th>Relative Score</th>
                <th style="text-align:right">Conf %</th>
              </tr>
              {bars_html}
            </table>

            <div class="disclaimer">
              ℹ️ Trained on the <em>20 Newsgroups</em> dataset (18,846 documents across 20 discussion
              categories). Confidence scores are derived from the model's decision function via softmax
              and represent relative certainty, not absolute probability.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"Input: {len(text.split())} words · {len(text)} characters")
