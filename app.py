# =========================================================
# AI POWERED PLAGIARISM DETECTION SYSTEM
# Mini / Second Year Project
# =========================================================


# ---------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# ---------------------------------------------------------

import streamlit as st              # For creating web application
import nltk                         # For NLP tasks
import string                       # To remove punctuation
import time                         # For progress animation
import base64, io                   # For background image handling
import PyPDF2                       # To read PDF files
from PIL import Image               # Image processing
from datetime import datetime       # For report date & time

# NLP utilities
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# AI / ML libraries
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# PDF report generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Plagiarism Detector",
    layout="wide"
)


# ---------------------------------------------------------
# DOWNLOAD REQUIRED NLTK DATA
# ---------------------------------------------------------
# Stopwords and sentence tokenizer are required for NLP

@st.cache_resource
def download_nltk_data():
    nltk.download("stopwords")
    nltk.download("punkt")

download_nltk_data()


# ---------------------------------------------------------
# BACKGROUND IMAGE LOADING & OPTIMIZATION
# ---------------------------------------------------------
# This function loads and compresses background image
# so that the web app loads faster

def load_background_image(path, max_width=1600):
    image = Image.open(path).convert("RGB")

    # Resize image if too large
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, int(image.height * ratio)))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70, optimize=True)

    # Convert image to base64 format
    return base64.b64encode(buffer.getvalue()).decode()


# ---------------------------------------------------------
# THEME SELECTION (LIGHT / DARK)
# ---------------------------------------------------------

theme = st.sidebar.radio("🌗 Theme", ["Dark", "Light"])

if theme == "Dark":
    background_image = load_background_image("bg_dark.jpg")
else:
    background_image = load_background_image("bg_light.jpg")


# ---------------------------------------------------------
# CUSTOM CSS FOR UI + MOBILE OPTIMIZATION
# ---------------------------------------------------------

st.markdown(f"""
<style>

/* Background with dark overlay for readability */
.stApp {{
    background-image:
        linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
        url("data:image/jpg;base64,{background_image}");
    background-size: cover;
    background-position: center;
}}

/* Headings */
h1,h2,h3,h4 {{
    color: #f8fafc !important;
    text-shadow: 0 2px 6px rgba(0,0,0,0.7);
}}

/* Text color */
p,label,span,div {{
    color: #e5e7eb !important;
}}

/* Glass card style */
.glass {{
    background: rgba(2,6,23,0.94);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.45);
    border: 1px solid rgba(255,255,255,0.06);
}}

/* Grade colors */
.green {{ color: #22c55e; font-weight: bold; }}
.yellow {{ color: #eab308; font-weight: bold; }}
.orange {{ color: #f97316; font-weight: bold; }}
.red {{ color: #ef4444; font-weight: bold; }}

/* Mobile optimization */
@media (max-width: 768px) {{
    h1 {{ font-size: 26px !important; }}
    h2 {{ font-size: 22px !important; }}
    h3 {{ font-size: 18px !important; }}

    .glass {{
        padding: 18px !important;
    }}

    button {{
        width: 100% !important;
        padding: 12px !important;
        font-size: 16px !important;
    }}
}}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD AI MODEL (BERT)
# ---------------------------------------------------------
# Using pre-trained sentence transformer model

@st.cache_resource
def load_ai_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

bert_model = load_ai_model()


# ---------------------------------------------------------
# TEXT PREPROCESSING FUNCTION
# ---------------------------------------------------------
# Removes unnecessary words and symbols

def preprocess_text(text):
    text = text.lower()  # convert to lowercase
    text = "".join(ch for ch in text if ch not in string.punctuation)

    words = text.split()
    filtered_words = [
        word for word in words
        if word not in stopwords.words("english")
    ]

    return " ".join(filtered_words)


# ---------------------------------------------------------
# PDF READING FUNCTION
# ---------------------------------------------------------

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    full_text = ""

    for page in reader.pages:
        if page.extract_text():
            full_text += page.extract_text()

    return full_text


# ---------------------------------------------------------
# PLAGIARISM GRADING FUNCTION
# ---------------------------------------------------------

def get_plagiarism_grade(score):
    if score <= 20:
        return "A", "green"
    elif score <= 40:
        return "B", "yellow"
    elif score <= 70:
        return "C", "orange"
    else:
        return "D", "red"


# ---------------------------------------------------------
# SENTENCE LEVEL PLAGIARISM CHECK
# ---------------------------------------------------------
# This function compares each sentence semantically

def sentence_level_comparison(text1, text2, threshold):
    sentences_1 = sent_tokenize(text1)
    sentences_2 = sent_tokenize(text2)

    embeddings_1 = bert_model.encode(sentences_1)
    embeddings_2 = bert_model.encode(sentences_2)

    results = []

    for index, emb in enumerate(embeddings_1):
        similarity_scores = cosine_similarity([emb], embeddings_2)[0]
        best_match_index = similarity_scores.argmax()
        similarity = similarity_scores[best_match_index] * 100

        if similarity >= threshold:
            results.append({
                "sentence_1": sentences_1[index],
                "sentence_2": sentences_2[best_match_index],
                "score": round(similarity, 2)
            })

    return results


# ---------------------------------------------------------
# PDF REPORT GENERATION FUNCTION
# ---------------------------------------------------------

def generate_pdf_report(similarity, grade, stats, matches):
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4)

    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("<b>AI Plagiarism Detection Report</b>", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Date: {datetime.now()}", styles["Normal"]))
    content.append(Paragraph(f"Similarity Score: {similarity:.2f}%", styles["Normal"]))
    content.append(Paragraph(f"Grade: {grade}", styles["Normal"]))
    content.append(Spacer(1, 12))

    table_data = [["Metric", "Value"]]
    for key, value in stats.items():
        table_data.append([key, str(value)])

    content.append(Table(table_data))
    content.append(Spacer(1, 12))

    for match in matches:
        content.append(
            Paragraph(
                f"<b>{match['score']}%</b><br/>"
                f"{match['sentence_1']}<br/>"
                f"Matched With: {match['sentence_2']}",
                styles["Normal"]
            )
        )
        content.append(Spacer(1, 10))

    document.build(content)
    buffer.seek(0)
    return buffer


# ---------------------------------------------------------
# APPLICATION TABS
# ---------------------------------------------------------

home_tab, analyze_tab, result_tab = st.tabs(
    ["🏠 Home", "🔍 Analyze", "📊 Results"]
)


# ---------------------------------------------------------
# HOME TAB CONTENT
# ---------------------------------------------------------

with home_tab:
    st.markdown("""
    <div class="glass" style="text-align:center;">
        <h1>AI-Powered Plagiarism Detection System</h1>
        <p>
        This system detects copied and paraphrased content using
        <b>Deep Learning, NLP and Explainable AI</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# ANALYZE TAB CONTENT
# ---------------------------------------------------------

with analyze_tab:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    file_1 = st.file_uploader("📄 Upload Document 1 (PDF / TXT)", ["pdf", "txt"])
    file_2 = st.file_uploader("📄 Upload Document 2 (PDF / TXT)", ["pdf", "txt"])

    threshold = st.slider("Similarity Threshold (%)", 50, 90, 75, 5)

    analyze_button = st.button("🔍 Analyze Plagiarism")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# RESULTS TAB LOGIC
# ---------------------------------------------------------

if analyze_button and file_1 and file_2:

    progress = st.progress(0)
    for i in range(0, 101, 10):
        time.sleep(0.03)
        progress.progress(i)

    # Read files
    text_1 = extract_text_from_pdf(file_1) if file_1.type == "application/pdf" else file_1.read().decode()
    text_2 = extract_text_from_pdf(file_2) if file_2.type == "application/pdf" else file_2.read().decode()

    # Calculate similarity
    similarity_score = cosine_similarity(
        bert_model.encode([preprocess_text(text_1)]),
        bert_model.encode([preprocess_text(text_2)])
    )[0][0] * 100

    grade, color_class = get_plagiarism_grade(similarity_score)
    matches = sentence_level_comparison(text_1, text_2, threshold)

    stats = {
        "Words in Document 1": len(text_1.split()),
        "Words in Document 2": len(text_2.split()),
        "Plagiarized Sentences": len(matches)
    }

    with result_tab:
        st.markdown(f"""
        <div class="glass" style="text-align:center;">
            <h1>{similarity_score:.2f}%</h1>
            <h3 class="{color_class}">Grade {grade}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Document Statistics")
        st.json(stats)

        st.markdown("### 🔎 Sentence-Level Matches")
        for index, match in enumerate(matches, 1):
            with st.expander(f"Match {index} - {match['score']}%"):
                st.write("Sentence 1:", match["sentence_1"])
                st.write("Matched Sentence:", match["sentence_2"])

        pdf_file = generate_pdf_report(similarity_score, grade, stats, matches)
        st.download_button("📄 Download Plagiarism Report", pdf_file, "plagiarism_report.pdf")


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("""
<div style="text-align:center; padding:30px; opacity:0.85;">
<hr>
<p>AI Plagiarism Detection System</p>
<p>Mini Project | Python • NLP • Deep Learning</p>
</div>
""", unsafe_allow_html=True)

