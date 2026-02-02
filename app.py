# ================== IMPORTS ==================
import streamlit as st
import nltk
import string
import time
import base64
import requests
import PyPDF2
from PIL import Image
import io

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_lottie import st_lottie


# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="AI Plagiarism Detector Pro",
    page_icon="🧠",
    layout="wide"
)


# ================== SAFE NLTK DOWNLOAD ==================
@st.cache_resource
def download_nltk():
    nltk.download("stopwords")
    nltk.download("punkt")

download_nltk()


# ================== IMAGE OPTIMIZATION ==================
def load_and_optimize_image(path, max_width=1600, quality=70):
    img = Image.open(path).convert("RGB")
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()


# ================== SIDEBAR (THEME ONLY) ==================
st.sidebar.title("🎨 Appearance")
theme = st.sidebar.radio("Theme", ["Dark", "Light"])

if theme == "Dark":
    bg_path = "bg_dark.jpg"
else:
    bg_path = "bg_light.jpg"

bg_image = load_and_optimize_image(bg_path)


# ================== BACKGROUND + READABILITY FIX ==================
st.markdown(f"""
<style>

/* ===== Background with fixed dark overlay ===== */
.stApp {{
    background-image:
        linear-gradient(
            rgba(0, 0, 0, 0.65),
            rgba(0, 0, 0, 0.65)
        ),
        url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* ===== Headings ===== */
h1, h2, h3, h4 {{
    color: #f8fafc !important;
    text-shadow: 0 2px 6px rgba(0,0,0,0.7);
}}

/* ===== General text ===== */
p, label, span, div {{
    color: #e5e7eb !important;
}}

/* ===== Glass cards (NO BLUR) ===== */
.glass {{
    background: rgba(2, 6, 23, 0.94);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 15px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.45);
    border: 1px solid rgba(255,255,255,0.06);
    transition: transform 0.3s ease;
}}

.glass:hover {{
    transform: scale(1.02);
}}

</style>
""", unsafe_allow_html=True)


# ================== HEADER + LOTTIE ==================
col1, col2 = st.columns([2,1])

with col1:
    st.markdown("<h1>🧠 AI-Powered Plagiarism Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Deep Learning • NLP • Explainable AI</h3>", unsafe_allow_html=True)

with col2:
    lottie_url = "https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json"
    st_lottie(requests.get(lottie_url).json(), height=220)

st.divider()


# ================== CORE FUNCTIONS ==================
def preprocess_text(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


@st.cache_resource(show_spinner=False)
def load_bert():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

model = load_bert()


def plagiarism_grade(score):
    if score <= 20:
        return "🟢 Original (Grade A)"
    elif score <= 40:
        return "🟡 Low (Grade B)"
    elif score <= 70:
        return "🟠 Medium (Grade C)"
    else:
        return "🔴 High (Grade D)"


def status_badge(score):
    if score <= 20:
        return "🟢 Original Content"
    elif score <= 40:
        return "🟡 Low Plagiarism"
    elif score <= 70:
        return "🟠 Moderate Plagiarism"
    else:
        return "🔴 High Plagiarism"


def sentence_wise_plagiarism(text1, text2, threshold):
    s1 = sent_tokenize(text1)
    s2 = sent_tokenize(text2)
    emb1 = model.encode(s1)
    emb2 = model.encode(s2)

    results = []
    for i, e1 in enumerate(emb1):
        scores = cosine_similarity([e1], emb2)[0]
        idx = scores.argmax()
        score = scores[idx] * 100
        if score >= threshold:
            results.append({
                "s1": s1[i],
                "s2": s2[idx],
                "score": round(score, 2)
            })
    return results


def top_k_sentence_matches(text1, text2, k=3):
    s1 = sent_tokenize(text1)
    s2 = sent_tokenize(text2)
    emb1 = model.encode(s1)
    emb2 = model.encode(s2)

    pairs = []
    for i, e1 in enumerate(emb1):
        scores = cosine_similarity([e1], emb2)[0]
        idx = scores.argmax()
        pairs.append({
            "s1": s1[i],
            "s2": s2[idx],
            "score": round(scores[idx] * 100, 2)
        })

    pairs.sort(key=lambda x: x["score"], reverse=True)
    return pairs[:k]


# ================== FILE UPLOAD ==================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("📁 Upload Documents")
c1, c2 = st.columns(2)
with c1:
    file1 = st.file_uploader("Document 1 (PDF / TXT)", type=["pdf","txt"])
with c2:
    file2 = st.file_uploader("Document 2 (PDF / TXT)", type=["pdf","txt"])
st.markdown('</div>', unsafe_allow_html=True)


# ================== SETTINGS ==================
st.markdown('<div class="glass">', unsafe_allow_html=True)
threshold = st.slider("Sentence Similarity Threshold (%)", 50, 90, 75, 5)
st.markdown('</div>', unsafe_allow_html=True)


# ================== ANALYZE ==================
if st.button("🔍 Analyze Plagiarism"):
    if not file1 or not file2:
        st.warning("Please upload both documents.")
    else:
        bar = st.progress(0)
        for i in range(0, 101, 10):
            time.sleep(0.04)
            bar.progress(i)

        text1 = read_pdf(file1) if file1.type=="application/pdf" else file1.read().decode("utf-8","ignore")
        text2 = read_pdf(file2) if file2.type=="application/pdf" else file2.read().decode("utf-8","ignore")

        similarity = cosine_similarity(
            model.encode([preprocess_text(text1)]),
            model.encode([preprocess_text(text2)])
        )[0][0] * 100

        grade = plagiarism_grade(similarity)

        tab1, tab2, tab3 = st.tabs(["📊 Overview","🔍 Analysis","🧠 Sentence Matches"])

        with tab1:
            st.markdown(f"""
            <div class="glass">
                <h2>{similarity:.2f}%</h2>
                <h4>{grade}</h4>
            </div>
            """, unsafe_allow_html=True)

            st.info(status_badge(similarity))
            st.progress(int(similarity))

            st.markdown("### 🔝 Top 3 Similar Sentences")
            for m in top_k_sentence_matches(text1, text2):
                st.markdown(f"""
                <div class="glass">
                    <b>{m['score']}%</b><br>
                    <i>{m['s1']}</i>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            with st.expander("📄 Document Statistics"):
                st.write("Document 1 words:", len(text1.split()))
                st.write("Document 2 words:", len(text2.split()))
                st.write("Threshold:", threshold, "%")

        with tab3:
            results = sentence_wise_plagiarism(text1, text2, threshold)
            if results:
                for r in results:
                    st.markdown(f"""
                    <div class="glass">
                        <b>{r['score']}%</b><br><br>
                        <b>Sentence:</b><br><i>{r['s1']}</i><br><br>
                        <b>Matched:</b><br><i>{r['s2']}</i>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No sentence-level plagiarism detected.")


st.divider()
st.markdown("<p style='text-align:center;'>🚀 AI-Powered Plagiarism Detection System</p>", unsafe_allow_html=True)
