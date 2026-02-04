from __future__ import annotations

from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd

# Pastikan module src.inference tersedia di environment Anda
try:
    from src.inference import load_tflite_model, predict_proba, top_k
except ImportError:
    st.error("Module 'src.inference' tidak ditemukan. Pastikan file inference.py ada di folder src.")
    st.stop()

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="FaunaVision | Animal Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS untuk Tampilan Modern ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .prediction-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 20px;
    }
    .intro-text {
        font-size: 1.1rem;
        color: #4a4a4a;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "tflite" / "model.tflite"
LABELS_PATH = APP_DIR / "tflite" / "label.txt"

@st.cache_resource
def get_model():
    return load_tflite_model(MODEL_PATH, LABELS_PATH)

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.image("https://www.tensorflow.org/images/tf_logo.png", width=100)
        st.title("Settings")
        st.subheader("Konfigurasi Model")
        topk = st.slider("Jumlah Prediksi Teratas (Top-K)", min_value=1, max_value=5, value=3)
        
        st.divider()
        st.info("""
        **Tips:** Gunakan foto yang fokus pada objek hewan dengan pencahayaan yang baik untuk akurasi maksimal.
        """)

    # --- Header & Pendahuluan ---
    st.title("🧠 FaunaVision: Klasifikasi Hewan")
    
    col_intro, col_model_info = st.columns([2, 1])
    
    with col_intro:
        st.markdown(f"""
        <div class="intro-text">
        Selamat datang di <b>FaunaVision</b>. Aplikasi ini menggunakan teknologi <i>Deep Learning</i> 
        untuk mengidentifikasi berbagai jenis spesies hewan secara instan. 
        Cukup unggah foto atau gunakan kamera Anda untuk memulai deteksi.
        </div>
        """, unsafe_allow_html=True)
        
    with col_model_info:
        with st.expander("ℹ️ Tentang Model TFLite"):
            st.write("""
            **TensorFlow Lite (TFLite)** adalah framework open-source untuk inferensi machine learning 
            pada perangkat seluler dan edge. 
            
            - **Efisiensi:** Model telah dikompresi agar berjalan cepat.
            - **Low Latency:** Inferensi dilakukan secara lokal tanpa perlu mengirim data ke server luar.
            - **Akurasi:** Menggunakan arsitektur saraf yang dioptimalkan untuk pengenalan gambar.
            """)

    st.divider()

    # --- Bagian Input ---
    tab1, tab2 = st.tabs(["📤 Unggah Gambar", "📸 Gunakan Kamera"])
    
    img = None
    with tab1:
        uploaded = st.file_uploader("Pilih file gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded)
            
    with tab2:
        camera = st.camera_input("Ambil foto hewan")
        if camera:
            img = Image.open(camera)

    # --- Logika Inferensi ---
    if img is not None:
        col_display, col_results = st.columns([1, 1], gap="large")
        
        with col_display:
            st.subheader("🖼️ Input Gambar")
            st.image(img, use_container_width=True, caption="Gambar yang akan dianalisis")

        with col_results:
            st.subheader("📊 Hasil Analisis")
            
            with st.spinner("Menganalisis pola gambar..."):
                model = get_model()
                proba = predict_proba(model, img)

                # Konversi ke Probabilitas (Softmax jika diperlukan)
                proba = np.asarray(proba, dtype=np.float32)
                if not (0.98 <= float(np.sum(proba)) <= 1.02):
                    e = np.exp(proba - np.max(proba))
                    proba = e / np.sum(e)

                results = top_k(proba, model.labels, k=topk)
                best_label, best_score = results[0]

            # Display Top-1 Metric
            st.markdown(f"""
            <div class="prediction-card">
                <h3>Prediksi Utama:</h3>
                <h1 style='color: #4CAF50;'>{best_label}</h1>
                <p>Tingkat Keyakinan: <b>{best_score:.2%}</b></p>
            </div>
            """, unsafe_allow_html=True)

            # Display Bar Chart for Top-K
            st.write("**Distribusi Kemungkinan:**")
            chart_data = pd.DataFrame(results, columns=["Spesies", "Skor"])
            chart_data = chart_data.set_index("Spesies")
            st.bar_chart(chart_data)
            
            # Table View
            with st.expander("Lihat detail angka"):
                for lbl, score in results:
                    st.write(f"- {lbl}: `{score:.4f}`")

    else:
        st.info("Silakan unggah gambar atau gunakan kamera di atas untuk melihat prediksi.")

    # --- Footer ---
    st.divider()
    st.caption("FaunaVision v2.0 | Powered by TensorFlow Lite & Streamlit")

if __name__ == "__main__":
    main()
