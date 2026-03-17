import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lymphoma Disease Predictor",
    page_icon="🔬",
    layout="centered",
)

# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("random_forest_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ─── Class Labels (from notebook) ────────────────────────────────────────────
CATEGORIES = ["Pre", "Benign", "Pro", "Early"]

CLASS_INFO = {
    "Pre":    {"color": "#3B82F6", "emoji": "🟦", "desc": "Pre-stage Lymphoma"},
    "Benign": {"color": "#22C55E", "emoji": "🟩", "desc": "Benign (Non-cancerous)"},
    "Pro":    {"color": "#F97316", "emoji": "🟧", "desc": "Prolymphocytic Lymphoma"},
    "Early":  {"color": "#EF4444", "emoji": "🟥", "desc": "Early-stage Lymphoma"},
}

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .main-header p {
        color: #6b7280;
        font-size: 1rem;
    }

    .upload-box {
        border: 2px dashed #6366f1;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: rgba(99, 102, 241, 0.04);
        transition: all 0.3s ease;
        margin: 1rem 0;
    }

    .result-card {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border-radius: 16px;
        padding: 1.8rem;
        color: white;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 8px 32px rgba(99,102,241,0.25);
        animation: fadeIn 0.6s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .result-card h2 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.4rem 0;
    }

    .result-card p {
        font-size: 1rem;
        opacity: 0.8;
        margin: 0;
    }

    .prob-label {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 0.2rem;
    }

    .disclaimer {
        background: #fef9c3;
        border-left: 4px solid #eab308;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #713f12;
        margin-top: 1.5rem;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 Lymphoma Disease Predictor</h1>
    <p>Upload a microscopy image to classify the Lymphoma subtype using Random Forest.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─── Sample images available locally ─────────────────────────────────────────
# Structure: samples/<Category>/1.png, 2.png, 3.png
SAMPLE_DIR = "samples"

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏷️ Classes")
    for cat, info in CLASS_INFO.items():
        st.markdown(f"{info['emoji']} **{cat}** — {info['desc']}")

    st.divider()
    st.markdown("### ⚡ Quick Demo")
    st.caption("Select a sample to test:")

    selected_sample = None
    
    for cat in CATEGORIES:
        st.markdown(f"**{CLASS_INFO[cat]['emoji']} {cat} Samples**")
        cols = st.columns(3)
        cat_dir = os.path.join(SAMPLE_DIR, cat)
        if os.path.exists(cat_dir):
            for i in range(1, 4):
                img_path = os.path.join(cat_dir, f"{i}.png")
                if os.path.exists(img_path):
                    with cols[i-1]:
                        # Load and show thumbnail
                        thumb = Image.open(img_path)
                        st.image(thumb, use_container_width=True)
                        if st.button(f"S{i}", key=f"btn_{cat}_{i}", use_container_width=True, help=f"Load {cat} sample {i}"):
                            selected_sample = img_path
        else:
            st.warning(f"No samples for {cat}")


# ─── File Upload ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a lymphoma microscopy image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    help="Supported formats: JPG, JPEG, PNG, BMP, TIF",
)

# Use sample if button was clicked and no file uploaded
if selected_sample and not uploaded_file:
    uploaded_file = open(selected_sample, "rb")

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    # ── Show uploaded image
    pil_image = Image.open(uploaded_file)
    with col1:
        st.markdown("**📷 Uploaded Image**")
        st.image(pil_image, use_container_width=True)

    # ── Preprocess: same as notebook
    img_array = np.array(pil_image.convert("RGB"))          # ensure 3 channels
    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # PIL is RGB; cv2 expects BGR
    img_resized = cv2.resize(img_bgr, (64, 64))
    img_flat    = img_resized.flatten()
    img_norm    = img_flat / 255.0
    img_input   = img_norm.reshape(1, -1)

    # ── Predict
    with st.spinner("Analysing image…"):
        pred_idx  = model.predict(img_input)[0]
        pred_label = CATEGORIES[pred_idx]
        info       = CLASS_INFO[pred_label]

        # Probability if the model supports it
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(img_input)[0]

    with col2:
        st.markdown("**🧬 Resized Preview (64×64)**")
        preview = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        st.image(preview, width=160)

    # ── Result card
    st.markdown(f"""
    <div class="result-card">
        <p>Predicted Class</p>
        <h2>{info['emoji']} {pred_label}</h2>
        <p>{info['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability bars
    if proba is not None:
        st.markdown("#### 📊 Class Probabilities")
        cols = st.columns(len(CATEGORIES))
        for i, (cat, col) in enumerate(zip(CATEGORIES, cols)):
            with col:
                pct = proba[i] * 100
                ci  = CLASS_INFO[cat]
                col.metric(label=f"{ci['emoji']} {cat}", value=f"{pct:.1f}%")
        st.progress(float(proba[pred_idx]))

    # ── Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Disclaimer:</strong> This tool is for educational and research purposes only.
        It is <strong>not</strong> a substitute for professional medical diagnosis.
        Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("⬆️ Please upload an image to get started.", icon="🖼️")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Lymphoma Subtype Classifier · Random Forest · BML Case Study")
