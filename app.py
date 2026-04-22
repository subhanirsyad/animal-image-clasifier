from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.inference import load_tflite_model, predict_proba, top_k


APP_DIR   = Path(__file__).parent
MODEL_PATH  = APP_DIR / "tflite" / "model.tflite"
LABELS_PATH = APP_DIR / "tflite" / "label.txt"

ANIMAL_META = {
    "butterfly": {"color": "#EC4899"},
    "cat":       {"color": "#F97316"},
    "chicken":   {"color": "#EAB308"},
    "cow":       {"color": "#22C55E"},
    "dog":       {"color": "#3B82F6"},
    "elephant":  {"color": "#8B5CF6"},
    "horse":     {"color": "#F59E0B"},
    "sheep":     {"color": "#14B8A6"},
    "spider":    {"color": "#EF4444"},
    "squirrel":  {"color": "#F97316"},
}
DEFAULT_COLOR = "#64748B"


def get_color(label: str) -> str:
    return ANIMAL_META.get(label.lower(), {}).get("color", DEFAULT_COLOR)


st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── reset & base ── */
html, body { background: #F1F5F9; }
[data-testid="stAppViewContainer"] { background: #F1F5F9; }
[data-testid="stAppViewContainer"] *:not(.hero-title):not(.hero-sub):not(.hero-tag):not(.top1-label):not(.top1-score):not(.top1-sub) {
    color: #1E293B;
}

/* ── sidebar ── */
[data-testid="stSidebar"] { background: #1E293B; }
[data-testid="stSidebar"] * { color: #F1F5F9 !important; }
[data-testid="stSidebar"] h2 { color: #FFFFFF !important; font-size: 1.1rem; font-weight: 700; }
[data-testid="stSidebar"] hr { border-color: #334155; }
[data-testid="stSidebar"] .stCheckbox label { color: #CBD5E1 !important; font-size: 0.85rem; }
[data-testid="stSidebar"] .stCaption { color: #94A3B8 !important; }

/* slider styling */
[data-testid="stSidebar"] [data-testid="stSlider"] {
    background: #0F172A;
    border-radius: 10px;
    padding: 14px 16px;
    border: 1px solid #334155;
}
[data-testid="stSidebar"] [data-testid="stSlider"] label {
    color: #94A3B8 !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #475569 !important;
    font-size: 0.75rem !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: #6366F1 !important;
    border-color: #6366F1 !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] div[data-testid="stSliderThumbValue"] {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
}

/* ── hero ── */
.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 40%, #DB2777 100%);
    border-radius: 16px;
    padding: 36px 44px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(79,70,229,0.3);
}
.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    color: #FFFFFF !important;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 3px 12px;
    border-radius: 20px;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 900;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    margin: 0 0 8px;
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.hero-sub {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.82) !important;
    max-width: 520px;
    line-height: 1.5;
    margin: 0;
}
.hero-stats {
    display: flex;
    gap: 24px;
    margin-top: 20px;
}
.hero-stat-val {
    font-size: 1.4rem;
    font-weight: 800;
    color: #FFFFFF !important;
}
.hero-stat-lbl {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.65) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── panel ── */
.panel {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 28px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    border: 1px solid #E2E8F0;
}
.panel-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.panel-title {
    font-size: 1rem;
    font-weight: 700;
    color: #0F172A !important;
    margin: 0;
}
.panel-desc {
    font-size: 0.8rem;
    color: #64748B !important;
    margin: 0 0 18px;
    line-height: 1.45;
}
.dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

/* ── upload area override ── */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 2px dashed #CBD5E1;
    border-radius: 12px;
    padding: 8px;
}
[data-testid="stFileUploader"] * { color: #475569 !important; }
[data-testid="stBaseButton-secondary"] {
    background: #334155 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
}
[data-testid="stBaseButton-secondary"]:hover { background: #1E293B !important; }

/* ── top-1 card ── */
.top1-card {
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.top1-label {
    font-size: 1.7rem;
    font-weight: 900;
    color: #FFFFFF !important;
    text-transform: capitalize;
    margin: 0 0 4px;
    text-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
.top1-score {
    font-size: 2.2rem;
    font-weight: 900;
    color: #FFFFFF !important;
    text-shadow: 0 1px 4px rgba(0,0,0,0.15);
}
.top1-sub {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.75) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* ── bar chart ── */
.bar-row { margin-bottom: 16px; }
.bar-meta { display: flex; justify-content: space-between; margin-bottom: 6px; }
.bar-name { font-weight: 700; font-size: 0.88rem; text-transform: capitalize; color: #1E293B !important; }
.bar-pct  { font-weight: 800; font-size: 0.88rem; color: #334155 !important; }
.bar-track { background: #E2E8F0; border-radius: 6px; height: 10px; overflow: hidden; }
.bar-fill  { height: 10px; border-radius: 6px; }

/* ── placeholder ── */
.placeholder {
    border: 2px dashed #E2E8F0;
    border-radius: 12px;
    padding: 44px 20px;
    text-align: center;
}
.placeholder p { color: #94A3B8 !important; font-size: 0.85rem; margin: 6px 0 0; }

/* ── sidebar sections ── */
.sb-section-title {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #94A3B8 !important;
    margin-bottom: 10px;
    margin-top: 4px;
}
.sb-step {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    margin-bottom: 12px;
}
.sb-step-num {
    background: #6366F1;
    color: #FFFFFF !important;
    font-size: 0.7rem;
    font-weight: 800;
    width: 20px; height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.sb-step-text { font-size: 0.82rem; color: #CBD5E1 !important; line-height: 1.4; }

.class-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 0;
    border-bottom: 1px solid #334155;
    font-size: 0.82rem;
    color: #CBD5E1 !important;
    text-transform: capitalize;
}
.class-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.sb-model-card {
    background: #0F172A;
    border-radius: 10px;
    padding: 14px 16px;
}
.sb-model-row { display: flex; justify-content: space-between; margin-bottom: 6px; }
.sb-model-key { font-size: 0.75rem; color: #64748B !important; }
.sb-model-val { font-size: 0.75rem; font-weight: 700; color: #94A3B8 !important; }

/* ── misc ── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
[data-testid="stImage"] img { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_tflite_model(MODEL_PATH, LABELS_PATH)


def render_bar(label: str, score: float):
    color = get_color(label)
    pct = score * 100
    st.markdown(f"""
    <div class="bar-row">
        <div class="bar-meta">
            <span class="bar-name">{label}</span>
            <span class="bar-pct">{pct:.1f}%</span>
        </div>
        <div class="bar-track">
            <div class="bar-fill" style="width:{max(pct,2):.1f}%; background:{color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(model):
    with st.sidebar:
        st.markdown("## Animal Classifier")
        st.markdown("---")

        # How to use
        st.markdown('<div class="sb-section-title">How to Use</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sb-step">
            <div class="sb-step-num">1</div>
            <div class="sb-step-text">Upload a clear photo of an animal or use your camera</div>
        </div>
        <div class="sb-step">
            <div class="sb-step-num">2</div>
            <div class="sb-step-text">The model analyzes the image automatically</div>
        </div>
        <div class="sb-step">
            <div class="sb-step-num">3</div>
            <div class="sb-step-text">View the predicted class and confidence scores</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Settings
        st.markdown('<div class="sb-section-title">Settings</div>', unsafe_allow_html=True)
        topk = st.slider("Top-K Results", min_value=1, max_value=5, value=3)
        use_camera = st.checkbox("Use Camera", value=False)

        st.markdown("---")

        # Supported classes
        st.markdown('<div class="sb-section-title">Supported Classes</div>', unsafe_allow_html=True)
        class_items = ""
        for label in model.labels:
            color = get_color(label)
            class_items += f"""
            <div class="class-item">
                <div class="class-dot" style="background:{color};"></div>
                {label}
            </div>"""
        st.markdown(class_items, unsafe_allow_html=True)

        st.markdown("---")

        # Model info
        st.markdown('<div class="sb-section-title">Model Info</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sb-model-card">
            <div class="sb-model-row">
                <span class="sb-model-key">Framework</span>
                <span class="sb-model-val">TensorFlow Lite</span>
            </div>
            <div class="sb-model-row">
                <span class="sb-model-key">Classes</span>
                <span class="sb-model-val">10 Animals</span>
            </div>
            <div class="sb-model-row" style="margin-bottom:0">
                <span class="sb-model-key">Input</span>
                <span class="sb-model-val">224 x 224 px</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    return topk, use_camera


def main():
    model = get_model()
    topk, use_camera = render_sidebar(model)

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-tag">Deep Learning · Image Classification</div>
        <h1 class="hero-title">Animal Image Classifier</h1>
        <p class="hero-sub">
            Upload a photo and let the model identify the animal.
            Powered by a TensorFlow Lite model trained on 10 animal categories
            with real-time inference directly in your browser.
        </p>
        <div class="hero-stats">
            <div>
                <div class="hero-stat-val">10</div>
                <div class="hero-stat-lbl">Animal Classes</div>
            </div>
            <div>
                <div class="hero-stat-val">TFLite</div>
                <div class="hero-stat-lbl">Model Format</div>
            </div>
            <div>
                <div class="hero-stat-val">224px</div>
                <div class="hero-stat-lbl">Input Size</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Left: input ──
    with col_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("""
        <div class="panel-head">
            <div class="dot" style="background:#6366F1;"></div>
            <p class="panel-title">Input Image</p>
        </div>
        <p class="panel-desc">
            Upload a JPG or PNG image with a clearly visible animal.
            For best results, use a centered, well-lit photo without heavy cropping.
        </p>
        """, unsafe_allow_html=True)

        img: Image.Image | None = None

        if use_camera:
            camera = st.camera_input("Take a photo")
            if camera is not None:
                img = Image.open(camera)
        else:
            uploaded = st.file_uploader(
                "Drop an image here, or click to browse",
                type=["jpg", "jpeg", "png"],
                label_visibility="visible",
            )
            if uploaded is not None:
                img = Image.open(uploaded)

        if img is not None:
            st.image(img, use_container_width=True)
        else:
            st.markdown("""
            <div class="placeholder" style="margin-top:12px;">
                <svg width="40" height="40" fill="none" stroke="#CBD5E1" stroke-width="1.5"
                     viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <rect x="3" y="3" width="18" height="18" rx="3"/>
                  <circle cx="8.5" cy="8.5" r="1.5"/>
                  <path d="M21 15l-5-5L5 21"/>
                </svg>
                <p>No image uploaded yet</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Right: results ──
    with col_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("""
        <div class="panel-head">
            <div class="dot" style="background:#22C55E;"></div>
            <p class="panel-title">Prediction Results</p>
        </div>
        <p class="panel-desc">
            The model outputs a confidence score for each class.
            The top prediction is shown prominently, followed by a ranked breakdown.
        </p>
        """, unsafe_allow_html=True)

        if img is None:
            st.markdown("""
            <div class="placeholder">
                <svg width="40" height="40" fill="none" stroke="#CBD5E1" stroke-width="1.5"
                     viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="11" cy="11" r="8"/>
                  <path d="M21 21l-4.35-4.35"/>
                </svg>
                <p>Upload an image on the left to see results</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Running inference..."):
                proba = predict_proba(model, img)
                proba = np.asarray(proba, dtype=np.float32)
                s = float(np.sum(proba))
                if not (0.98 <= s <= 1.02):
                    e = np.exp(proba - np.max(proba))
                    proba = e / np.sum(e)
                results = top_k(proba, model.labels, k=topk)

            best_label, best_score = results[0]
            color = get_color(best_label)

            st.markdown(f"""
            <div class="top1-card" style="background: linear-gradient(135deg, {color}dd, {color});">
                <p class="top1-sub">Top Prediction</p>
                <p class="top1-label">{best_label}</p>
                <p class="top1-score">{best_score:.1%}</p>
                <p class="top1-sub">confidence</p>
            </div>
            """, unsafe_allow_html=True)

            if len(results) > 1:
                st.markdown(
                    '<p style="font-size:0.78rem; font-weight:700; text-transform:uppercase; '
                    'letter-spacing:1px; color:#94A3B8; margin-bottom:14px;">All Predictions</p>',
                    unsafe_allow_html=True
                )
                for label, score in results:
                    render_bar(label, score)

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
