# ==========================================================
# AI PLAGIARISM DETECTOR PRO
# Human-friendly, explainable & demo-ready version
# ==========================================================

# ================== IMPORTS ==================
import streamlit as st
import nltk
import string
import time
import base64
import requests
import io

import PyPDF2
from PIL import Image

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_lottie import st_lottie


# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="AI Plagiarism Detector Pro",
    layout="wide"
)


# ================== NLTK SETUP (SAFE & CACHED) ==================
@st.cache_resource
def setup_nltk():
    """
    Download required NLTK resources only once.
    Prevents repeated downloads on every app reload.
    """
    nltk.download("stopwords")
    nltk.download("punkt")

setup_nltk()


# ================== IMAGE OPTIMIZATION ==================
def load_background_image(path, max_width=1600, quality=70):
    """
    Loads background image, resizes it for performance,
    and converts it into base64 for Streamlit CSS usage.
    """
    img = Image.open(path).convert("RGB")

    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()


# ================== SIDEBAR : THEME SELECTION ==================
st.sidebar.title("🎨 Appearance Settings")

theme = st.sidebar.radio(
    "Choose app theme",
    ["Dark", "Light"],
    help="Change background style for better readability"
)

bg_path = "bg_dark.jpg" if theme == "Dark" else "bg_light.jpg"
bg_image = load_background_image(bg_path)


# ================== GLOBAL STYLING ==================
st.markdown(f"""
<style>

/* App background with readability overlay */
.stApp {{
    background-image:
        linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
        url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* Headings */
h1, h2, h3, h4 {{
    color: #f8fafc !important;
    text-shadow: 0 2px 6px rgba(0,0,0,0.7);
}}

/* Text */
p, label, span, div {{
    color: #e5e7eb !important;
}}

/* Glass-style cards */
.glass {{
    background: rgba(2, 6, 23, 0.94);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 16px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.45);
    border: 1px solid rgba(255,255,255,0.06);
    transition: transform 0.3s ease;
}}

.glass:hover {{
    transform: scale(1.02);
}}

</style>
""", unsafe_allow_html=True)


# ================== HEADER SECTION ==================
left, right = st.columns([2, 1])

with left:
    st.markdown("<h1>AI-Powered Plagiarism Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Semantic Similarity • NLP • Deep Learning</h3>", unsafe_allow_html=True)
    st.markdown(
        "Upload two documents and let AI intelligently check **how similar they really are**."
    )

with right:
    lottie_url = "https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json"
    st_lottie(requests.get(lottie_url).json(), height=220)

st.divider()


# ================== CORE NLP UTILITIES ==================
def preprocess_text(text):
    """
    Cleans text for better semantic comparison:
    - Lowercase
    - Remove punctuation
    - Remove stopwords
    """
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)

    words = text.split()
    filtered_words = [w for w in words if w not in stopwords.words("english")]

    return " ".join(filtered_words)


def extract_text(file):
    """
    Reads text from PDF or TXT files safely.
    """
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text
    else:
        return file.read().decode("utf-8", "ignore")


# ================== AI MODEL LOADING ==================
@st.cache_resource(show_spinner=False)
def load_ai_model():
    """
    Loads sentence transformer model only once.
    """
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

model = load_ai_model()


# ================== RESULT INTERPRETATION ==================
def plagiarism_grade(score):
    if score <= 20:
        return "🟢 Grade A – Original Content"
    elif score <= 40:
        return "🟡 Grade B – Minor Similarity"
    elif score <= 70:
        return "🟠 Grade C – Moderate Plagiarism"
    else:
        return "🔴 Grade D – High Plagiarism"


def status_badge(score):
    if score <= 20:
        return "Content appears original."
    elif score <= 40:
        return "Low similarity detected."
    elif score <= 70:
        return "Moderate similarity detected."
    else:
        return "High plagiarism detected."


# ================== SENTENCE LEVEL ANALYSIS ==================
def sentence_level_matches(text1, text2, threshold):
    s1 = sent_tokenize(text1)
    s2 = sent_tokenize(text2)

    emb1 = model.encode(s1)
    emb2 = model.encode(s2)

    matches = []

    for i, e1 in enumerate(emb1):
        scores = cosine_similarity([e1], emb2)[0]
        best_idx = scores.argmax()
        similarity = scores[best_idx] * 100

        if similarity >= threshold:
            matches.append({
                "sentence_1": s1[i],
                "sentence_2": s2[best_idx],
                "score": round(similarity, 2)
            })

    return matches


# ================== FILE UPLOAD SECTION ==================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("📂 Upload Your Documents")

c1, c2 = st.columns(2)
with c1:
    file1 = st.file_uploader("Document 1", type=["pdf", "txt"])
with c2:
    file2 = st.file_uploader("Document 2", type=["pdf", "txt"])

st.markdown('</div>', unsafe_allow_html=True)


# ================== ANALYSIS SETTINGS ==================
st.markdown('<div class="glass">', unsafe_allow_html=True)
threshold = st.slider(
    "Sentence Similarity Threshold (%)",
    50, 90, 75, 5,
    help="Higher value = stricter plagiarism detection"
)
st.markdown('</div>', unsafe_allow_html=True)


# ================== MAIN ANALYSIS ==================
if st.button("🔍 Analyze Plagiarism"):

    if not file1 or not file2:
        st.warning("Please upload both documents to continue.")
    else:
        progress = st.progress(0)
        for i in range(0, 101, 10):
            time.sleep(0.04)
            progress.progress(i)

        text1 = extract_text(file1)
        text2 = extract_text(file2)

        similarity_score = cosine_similarity(
            model.encode([preprocess_text(text1)]),
            model.encode([preprocess_text(text2)])
        )[0][0] * 100

        tabs = st.tabs(["📊 Overview", "📄 Details", "🧠 Sentence Matches"])

        with tabs[0]:
            st.markdown(f"""
            <div class="glass">
                <h2>{similarity_score:.2f}% Similarity</h2>
                <h4>{plagiarism_grade(similarity_score)}</h4>
            </div>
            """, unsafe_allow_html=True)

            st.info(status_badge(similarity_score))
            st.progress(int(similarity_score))

        with tabs[1]:
            st.write("📄 Document 1 words:", len(text1.split()))
            st.write("📄 Document 2 words:", len(text2.split()))
            st.write("🎯 Threshold:", threshold, "%")

        with tabs[2]:
            matches = sentence_level_matches(text1, text2, threshold)
            if matches:
                for m in matches:
                    st.markdown(f"""
                    <div class="glass">
                        <b>{m['score']}%</b><br><br>
                        <b>Sentence:</b><br><i>{m['sentence_1']}</i><br><br>
                        <b>Matched With:</b><br><i>{m['sentence_2']}</i>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No significant sentence-level plagiarism detected.")


st.divider()
st.markdown(
    "<p style='text-align:center;'>AI-Powered Plagiarism Detection System • Built using NLP & Deep Learning</p>",
    unsafe_allow_html=True
)
