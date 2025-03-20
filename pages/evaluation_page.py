import streamlit as st
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentiment_analyzer import SentimentAnalyzer
from preprocess import TextPreprocessor

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Model Evaluation", layout="wide")

# Path ke folder model
MODEL_DIR = "D:/SKRIPSI/UI Streamlit/model/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kamus model dengan informasi konfigurasi
MODEL_CONFIGS = {
    "bert_model1.pth": {"lr": "2e-5", "epoch": 5},
    "bert_model2.pth": {"lr": "2e-5", "epoch": 10},
    "bert_model3.pth": {"lr": "2e-5", "epoch": 15},
    "bert_model4.pth": {"lr": "2e-5", "epoch": 20},
    "bert_model5.pth": {"lr": "2e-6", "epoch": 5},
    "bert_model6.pth": {"lr": "2e-6", "epoch": 10},
    "bert_model7.pth": {"lr": "2e-6", "epoch": 15},
    "bert_model8.pth": {"lr": "2e-6", "epoch": 20},
}


# Fungsi untuk mengubah nama model menjadi lebih informatif
def format_model_name(model_filename):
    config = MODEL_CONFIGS.get(model_filename, {})
    return f"🔹 Model {model_filename} | LR: {config['lr']} | Epochs: {config['epoch']}"


# Ambil daftar model yang tersedia dalam folder
available_models = [f for f in os.listdir(MODEL_DIR) if f in MODEL_CONFIGS]
model_display_names = {format_model_name(m): m for m in available_models}

# **BAGIAN 1: Pemilihan Model**
st.title("📊 Model Evaluation")
st.subheader("🔍 Pilih Model untuk Evaluasi")

selected_display_name = st.selectbox("Pilih Model:", list(model_display_names.keys()))
selected_model_file = model_display_names[selected_display_name]  # Ambil nama file asli

# Tampilkan detail model sebelum tombol ditekan
selected_config = MODEL_CONFIGS[selected_model_file]
st.write(f"**Detail Model yang Dipilih:**")
st.write(f"📌 **Nama File:** {selected_model_file}")
st.write(f"⚡ **Learning Rate:** {selected_config['lr']}")
st.write(f"🕒 **Epochs:** {selected_config['epoch']}")

# **Tombol konfirmasi untuk memuat model**
if st.button("✅ Muat Model"):
    analyzer = SentimentAnalyzer(os.path.join(MODEL_DIR, selected_model_file))
    preprocessor = TextPreprocessor()
    st.session_state.analyzer = analyzer
    st.session_state.preprocessor = preprocessor
    st.session_state.model_loaded = True
    st.success(f"✅ Model {selected_display_name} berhasil dimuat!")

# **Periksa apakah model sudah dimuat**
if "model_loaded" in st.session_state and st.session_state.model_loaded:
    analyzer = st.session_state.analyzer
    preprocessor = st.session_state.preprocessor

    # **BAGIAN 2: Upload dan Evaluasi Dataset**
    st.subheader("📂 Upload Dataset untuk Evaluasi")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "text" in df.columns and "label" in df.columns:
            st.write("🔍 **Preview Data:**")
            st.write(df.head())

            # **Lakukan prapemrosesan teks menggunakan TextPreprocessor**
            df["clean_text"] = df["text"].apply(preprocessor.clean_text)

            # **Lakukan prediksi**
            y_true = df["label"].values
            y_pred = [analyzer.predict(text)[0] for text in df["clean_text"]]

            # Konversi label prediksi ke angka (0 untuk negatif, 1 untuk positif)
            y_pred_numeric = [
                1 if sentiment == "😊 Positive" else 0 for sentiment in y_pred
            ]

            # **Evaluasi Model**
            accuracy = accuracy_score(y_true, y_pred_numeric)
            report = classification_report(y_true, y_pred_numeric, output_dict=True)
            cm = confusion_matrix(y_true, y_pred_numeric)

            # **Tampilkan hasil dalam 2 kolom**
            col1, col2 = st.columns([1.5, 1])

            with col1:
                # **Tampilkan classification report dalam tabel**
                st.markdown(f"### ✅ **Accuracy: {accuracy:.4f}**")
                st.write("### 📋 Classification Report:")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.4f}"))

            with col2:
                # **Tampilkan confusion matrix dalam ukuran optimal**
                st.write("### 🔄 Confusion Matrix:")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"],
                )
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.error("CSV file harus memiliki kolom 'text' dan 'label'.")
else:
    st.warning(
        "⚠️ Harap pilih dan muat model terlebih dahulu sebelum mengevaluasi dataset."
    )
