import streamlit as st
from sentiment_analyzer import SentimentAnalyzer
from preprocess import TextPreprocessor

# Warna Spotify
SPOTIFY_GREEN = "#1DB954"
NEGATIVE_RED = "#FF4C4C"
SPOTIFY_BLACK = "#191414"

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Spotify Sentiment Analysis", layout="wide")

# Tambahkan gaya CSS untuk tampilan lebih menarik
st.markdown(
    f"""
    <style>
        body {{ background-color: {SPOTIFY_BLACK}; color: white; }}
        .stTextArea textarea {{ background-color: white; color: black; }}
        .center-content {{ display: flex; justify-content: center; align-items: center; }}
        .stButton>button {{
            background-color: {SPOTIFY_GREEN}; color: white;
            font-size: 16px; font-weight: bold;
            border-radius: 10px; padding: 10px 20px;
        }}
        .stButton>button:hover {{ background-color: #17A347; }}
        .result-box {{
            width: 60%; min-width: 300px; max-width: 600px;
            padding: 25px; border-radius: 15px; text-align: center;
            color: black; margin-top: 15px; font-size: 18px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# Pemuatan model dan preprocessor hanya sekali menggunakan cache
@st.cache_resource
def load_resources():
    model_path = "D:/SKRIPSI/UI Streamlit/model/bert_model6.pth"
    analyzer = SentimentAnalyzer(model_path)
    preprocessor = TextPreprocessor()
    return analyzer, preprocessor


analyzer, preprocessor = load_resources()

st.markdown(
    f"""
    <h1 style='text-align: center; color: {SPOTIFY_GREEN};'>🎵 Spotify Sentiment Analysis 🎵</h1>
    <h3 style='text-align: center;'>Masukkan ulasan tentang Spotify dan lihat analisis sentimennya!</h3>
    """,
    unsafe_allow_html=True,
)

# Input teks dari user
user_input = st.text_area("Masukkan ulasan Anda di sini:", height=150)

# **Tombol Analisis Sentimen**
st.markdown('<div class="center-content">', unsafe_allow_html=True)
analyze_button = st.button("🎧 Analisis Sentimen")
st.markdown("</div>", unsafe_allow_html=True)

# **Hasil Prediksi**
if analyze_button:
    if user_input.strip():
        # Preprocess teks sebelum dianalisis
        cleaned_text = preprocessor.clean_text(user_input)
        sentiment, prob_neg, prob_pos = analyzer.predict(cleaned_text)

        # Tentukan warna hasil berdasarkan sentimen
        box_color = SPOTIFY_GREEN if sentiment == "😊 Positive" else NEGATIVE_RED

        st.markdown(
            f"""
            <div class="center-content">
                <div class="result-box" style="background-color: {box_color};">
                    <h2>{sentiment}</h2>
                    <p><b>Probabilitas Negatif:</b> {prob_neg:.4f}</p>
                    <p><b>Probabilitas Positif:</b> {prob_pos:.4f}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("❗ Harap masukkan teks ulasan terlebih dahulu.")
