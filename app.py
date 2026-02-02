from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.inference import load_tflite_model, predict_proba, top_k


APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "tflite" / "model.tflite"
LABELS_PATH = APP_DIR / "tflite" / "label.txt"


st.set_page_config(
    page_title="Animal Image Classifier (TFLite)",
    page_icon="🦋",
    layout="centered",
)


@st.cache_resource
def get_model():
    return load_tflite_model(MODEL_PATH, LABELS_PATH)


def main():
    st.title("🧠 Animal Image Classifier")
    st.caption("Upload an image and the app will predict the animal class using a TensorFlow Lite model.")

    with st.expander("Supported classes", expanded=False):
        model = get_model()
        st.write(", ".join(model.labels))

    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)
    with col1:
        use_camera = st.checkbox("Use camera", value=False)
    with col2:
        topk = st.slider("Top-K results", min_value=1, max_value=5, value=3)

    img: Image.Image | None = None

    if use_camera:
        camera = st.camera_input("Take a photo")
        if camera is not None:
            img = Image.open(camera)
    elif uploaded is not None:
        img = Image.open(uploaded)

    if img is None:
        st.info("Upload an image (or use camera) to get a prediction.")
        st.stop()

    st.subheader("Input")
    st.image(img, use_container_width=True)

    with st.spinner("Running inference..."):
        model = get_model()
        proba = predict_proba(model, img)

        # Some models may output logits; convert to probabilities if needed.
        # We detect this loosely: if values don't sum near 1, apply softmax.
        proba = np.asarray(proba, dtype=np.float32)
        s = float(np.sum(proba))
        if not (0.98 <= s <= 1.02):
            # softmax
            e = np.exp(proba - np.max(proba))
            proba = e / np.sum(e)

        results = top_k(proba, model.labels, k=topk)

    best_label, best_score = results[0]

    st.subheader("Prediction")
    st.metric("Top-1", best_label, f"{best_score:.2%}")

    st.write("**Top results:**")
    for lbl, score in results:
        st.write(f"- {lbl}: **{score:.2%}**")

    st.divider()
    st.caption("Tip: Try clear, centered animal photos for better results.")


if __name__ == "__main__":
    main()
