# Animal Image Classifier (Streamlit + TensorFlow Lite)

Aplikasi web sederhana untuk **klasifikasi gambar** menggunakan **Streamlit** yang menjalankan **inference TensorFlow Lite (TFLite)** untuk memprediksi kelas hewan dari foto yang kamu upload.

## Fitur

- UI Streamlit (`app.py`)
- Inference dengan model **TFLite** (`tflite/model.tflite`)
- Label kelas di `tflite/label.txt`
- Notebook training disertakan di `notebooks/training_notebook.ipynb`
- Struktur repo sudah rapi untuk langsung di-push ke GitHub

## Kelas yang didukung

Model memprediksi salah satu dari 10 kelas berikut:

`butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel`

## Struktur project

```text
.
├─ app.py
├─ requirements.txt
├─ requirements-dev.txt
├─ src/
│  └─ inference.py
├─ tflite/
│  ├─ model.tflite
│  └─ label.txt
├─ saved_model/
├─ tfjs_model/
└─ notebooks/
   └─ training_notebook.ipynb
```

## Cara menjalankan lokal

```bash
# 1) (Opsional) buat virtual env
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) install dependency
pip install -r requirements.txt

# 3) jalankan aplikasi
streamlit run app.py
```

Setelah itu buka URL yang muncul di terminal (biasanya `http://localhost:8501`).

## Deploy (biar dapat link publik sendiri)

Cara termudah: **Streamlit Community Cloud**.

1. Push repo ini ke GitHub.
2. Buka Streamlit Community Cloud → klik **Create app**.
3. Pilih repo GitHub kamu.
4. Set **Main file path**: `app.py`
5. Klik **Deploy** → Streamlit akan membuat **link publik** untuk aplikasimu (domain `*.streamlit.app`).

> Catatan: link publik baru ada setelah kamu deploy. README ini sudah siap dipakai di GitHub tanpa perlu kamu edit.

## Catatan input model

- Model export memiliki layer rescaling internal, jadi input dipakai sebagai **float32** dengan rentang pixel **0..255**.
- Gambar akan di-resize ke **224×224** sebelum inference.

## Lisensi

Repo ini belum menyertakan file lisensi. Kalau mau open-source, kamu bisa menambahkan lisensi (mis. MIT).
