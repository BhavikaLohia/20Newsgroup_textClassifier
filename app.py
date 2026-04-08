import streamlit as st
import pickle
import os

st.set_page_config(page_title="NewsGroup Classifier", page_icon="📰", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #F0F4FF !important;
    font-family: 'Inter', sans-serif !important;
    color: #1a1a2e !important;
}
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
#MainMenu, footer, header        { visibility: hidden; }
.block-container { padding-top: 2.5rem !important; max-width: 720px !important; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 48px 24px 36px;
}
.hero-icon {
    font-size: 3rem;
    margin-bottom: 12px;
    display: block;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1a1a2e;
    margin: 0 0 10px;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1rem;
    color: #6b7280;
    margin: 0;
    font-weight: 400;
}

/* ── Card wrapper ── */
.card {
    background: #ffffff;
    border-radius: 20px;
    padding: 32px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 8px 32px rgba(99,102,241,0.08);
    margin-bottom: 20px;
}
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 10px;
}

/* ── Textarea ── */
.stTextArea textarea {
    background: #F9FAFB !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    font-family: 'Inter', sans-serif !important;
    color: #1a1a2e !important;
    padding: 14px 16px !important;
    line-height: 1.6 !important;
    resize: none !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    background: #fff !important;
}
.stTextArea label { display: none !important; }

/* ── Button ── */
div.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 48px !important;
    font-size: 0.97rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.01em !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
    transition: all 0.2s !important;
}
div.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
    transform: translateY(-1px) !important;
}
div.stButton > button:active { transform: translateY(0) !important; }

/* ── Result ── */
.result-wrap {
    background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(99,102,241,0.30);
    margin-bottom: 20px;
}
.result-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    color: rgba(255,255,255,0.7);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 10px;
}
.result-icon { font-size: 2.4rem; margin-bottom: 8px; display: block; }
.result-name {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 14px;
    margin-bottom: 20px;
}
.stat-box {
    flex: 1;
    background: #ffffff;
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.stat-value { font-size: 1.5rem; font-weight: 800; color: #6366f1; }
.stat-label { font-size: 0.72rem; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }

/* ── Error ── */
.err-box {
    background: #fff7ed;
    border: 1.5px solid #fed7aa;
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: #c2410c;
    margin-top: 12px;
    text-align: center;
}

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 0.75rem;
    color: #d1d5db;
    margin-top: 32px;
    padding-bottom: 24px;
}
.footer span { color: #6366f1; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Label maps ────────────────────────────────────────────────────────────────
SIMPLE_NAMES = {
    "alt.atheism":               "Atheism",
    "comp.graphics":             "Computer Graphics",
    "comp.os.ms-windows.misc":   "Windows OS",
    "comp.sys.ibm.pc.hardware":  "IBM PC Hardware",
    "comp.sys.mac.hardware":     "Mac Hardware",
    "comp.windows.x":            "Windows X",
    "misc.forsale":              "For Sale",
    "rec.autos":                 "Cars",
    "rec.motorcycles":           "Motorcycles",
    "rec.sport.baseball":        "Baseball",
    "rec.sport.hockey":          "Hockey",
    "sci.crypt":                 "Cryptography",
    "sci.electronics":           "Electronics",
    "sci.med":                   "Medicine",
    "sci.space":                 "Space",
    "soc.religion.christian":    "Christianity",
    "talk.politics.guns":        "Guns & Politics",
    "talk.politics.mideast":     "Middle East Politics",
    "talk.politics.misc":        "Politics (Misc)",
    "talk.religion.misc":        "Religion (Misc)",
}

CATEGORY_ICONS = {
    "Atheism":                "🔮",
    "Computer Graphics":      "🖥️",
    "Windows OS":             "🪟",
    "IBM PC Hardware":        "💾",
    "Mac Hardware":           "🍎",
    "Windows X":              "⚙️",
    "For Sale":               "🏷️",
    "Cars":                   "🚗",
    "Motorcycles":            "🏍️",
    "Baseball":               "⚾",
    "Hockey":                 "🏒",
    "Cryptography":           "🔐",
    "Electronics":            "⚡",
    "Medicine":               "🩺",
    "Space":                  "🚀",
    "Christianity":           "✝️",
    "Guns & Politics":        "🗳️",
    "Middle East Politics":   "🌍",
    "Politics (Misc)":        "🏛️",
    "Religion (Misc)":        "🕊️",
}

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    missing = [f for f in ("model.pkl", "vectorizer.pkl", "labels.pkl") if not os.path.exists(f)]
    if missing:
        return None, None, None, missing
    with open("model.pkl",      "rb") as f: model      = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f: vectorizer = pickle.load(f)
    with open("labels.pkl",     "rb") as f: labels     = pickle.load(f)
    return model, vectorizer, labels, []

model, vectorizer, labels, missing_files = load_artifacts()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="hero-icon">📰</span>
  <h1>NewsGroup Classifier</h1>
  <p>Enter any text and the AI will instantly identify its topic category.</p>
</div>
""", unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
  <div class="stat-box">
    <div class="stat-value">20</div>
    <div class="stat-label">Categories</div>
  </div>
  <div class="stat-box">
    <div class="stat-value">LinearSVC</div>
    <div class="stat-label">Model</div>
  </div>
  <div class="stat-box">
    <div class="stat-value">TF-IDF</div>
    <div class="stat-label">Features</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Missing files warning ─────────────────────────────────────────────────────
if missing_files:
    st.markdown(f"""
    <div class="err-box">
      ⚠️ <strong>Missing files:</strong> {', '.join(missing_files)}<br>
      Place <code>model.pkl</code>, <code>vectorizer.pkl</code>, and <code>labels.pkl</code>
      in the same folder as this script, then restart.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Input card ────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="section-label">Input Text</div>', unsafe_allow_html=True)

text_input = st.text_area(
    "text",
    placeholder="Paste a news article, forum post, or any text here…",
    height=180,
    label_visibility="collapsed",
)

classify = st.button("✦ Classify Text")
st.markdown('</div>', unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
def predict_category(model, vectorizer, labels, text):
    vec = vectorizer.transform([text])
    pred_idx = model.predict(vec)[0]
    raw_label = labels[pred_idx]
    return SIMPLE_NAMES.get(raw_label, raw_label)

if classify:
    raw = text_input.strip()
    if not raw:
        st.markdown('<div class="err-box">⚠️ Please enter some text before classifying.</div>', unsafe_allow_html=True)
    elif len(raw.split()) < 3:
        st.markdown('<div class="err-box">⚠️ Text is too short — try a few more words.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Analysing text…"):
            category = predict_category(model, vectorizer, labels, raw)
        icon = CATEGORY_ICONS.get(category, "📄")
        st.markdown(f"""
        <div class="result-wrap">
          <div class="result-eyebrow">Predicted Category</div>
          <span class="result-icon">{icon}</span>
          <div class="result-name">{category}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">Built with <span>Streamlit</span> · 20 Newsgroups · TF-IDF + LinearSVC</div>', unsafe_allow_html=True)
