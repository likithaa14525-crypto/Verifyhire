import streamlit as st
import pickle
import re
import requests
import PyPDF2
from bs4 import BeautifulSoup
from lime.lime_text import LimeTextExplainer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="VerifyHire", page_icon="🛡️", layout="wide")

# ---------- Custom CSS for bold/impactful styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3 {
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}
.metric-card {
    background-color: #f5f3ff;
    border: 1px solid #ddd6fe;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #6d28d9;
}
.metric-label {
    font-size: 13px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.info-box {
    background-color: #ede9fe;
    border-radius: 10px;
    padding: 14px;
    font-size: 14px;
    color: #4c1d95;
}
.demo-card {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 8px;
    font-size: 13px;
}
.stButton>button {
    background-color: #7c3aed;
    color: white;
    font-weight: 700;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
}
.stButton>button:hover {
    background-color: #6d28d9;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load fraud detection model + vectorizer ----------
@st.cache_resource
def load_model():
    with open("model/logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ---------- Helpers ----------
def is_url(text):
    return bool(re.match(r"^https?://", text.strip()))

def scrape_job_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return None

def predict_proba(texts):
    tfidf = vectorizer.transform(texts)
    return model.predict_proba(tfidf)

# Quick demo templates
DEMO_TEMPLATES = {
    "🚨 Scam: Home-Based Data Entry": "URGENT: Home-based Data Entry Clerks needed! Pay is $35-$50 per hour. No experience required. Send your bank details and a $50 registration fee to start immediately.",
    "🚨 Scam: Financial Operations": "International Asset Group is seeking remote Financial processing agents. You will receive funds into your personal account and forward them via wire transfer. Earn 10% commission per transaction.",
    "✅ Real: Junior Frontend Developer": "We are a growing local e-commerce startup looking for a passionate Junior Frontend Developer. Requirements: HTML, CSS, JavaScript, React basics. Full-time, in-office role with standard benefits.",
    "✅ Real: Entry-Level Data Analyst": "Apex Analytics is hiring an Entry-Level Data Analyst. You will assist in cleaning, analyzing, and visualizing datasets using Python and SQL. Bachelor's degree in a related field required.",
}

if "job_text_value" not in st.session_state:
    st.session_state.job_text_value = ""

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🛡️ VerifyHire AI")

    st.markdown("""
    <div class="info-box">
    <b>Current Architecture</b><br>
    🔹 TF-IDF + Logistic Regression<br>
    🔹 LIME Explainer Engine
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Model Stats (EMSCAD)**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-value">97%</div>
        <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-value">90%</div>
        <div class="metric-label">Fake Recall</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>💡 Explainable AI Mode</b><br>
    Highlights the exact words driving each verdict, so job seekers instantly recognize employment scam patterns.
    </div>
    """, unsafe_allow_html=True)

# ---------- Main tabs ----------
tab1, tab2 = st.tabs(["🔍 Fake Job Detector", "🎯 Resume Match"])

# ======================================================
# TAB 1: Fake Job Detector
# ======================================================
with tab1:
    st.title("🔍 Explainable Job Fraud Classifier")
    st.write("Examine any job description text or a web page link. We break down which specific keywords are associated with employment scams.")

    st.markdown("**⚡ Quick Demo Templates (try these!)**")
    demo_cols = st.columns(4)
    for idx, (label, text) in enumerate(DEMO_TEMPLATES.items()):
        with demo_cols[idx]:
            if st.button(label, key=f"demo_{idx}"):
                st.session_state.job_text_value = text

    input_mode = st.radio("Input method:", ["Paste Job Description", "Analyze URL"], horizontal=True)

    if input_mode == "Paste Job Description":
        job_input = st.text_area(
            "Paste job description text:",
            value=st.session_state.job_text_value,
            height=180,
            key="job_text_area"
        )
    else:
        job_input = st.text_input("e.g. https://www.indeed.com/viewjob?jk=12345", key="job_url_input")

    if st.button("🔍 Detect Fraudulent Activity", key="analyze_btn"):
        if not job_input.strip():
            st.warning("Please paste some job text or a URL first.")
        else:
            job_text = job_input.strip()

            if is_url(job_text):
                with st.spinner("Scraping job posting from URL..."):
                    scraped = scrape_job_text(job_text)
                if scraped is None or len(scraped) < 50:
                    st.error("Could not scrape enough text from that URL (common with JavaScript-heavy sites like Indeed/LinkedIn). Try pasting the job text directly instead.")
                    st.stop()
                job_text = scraped

            with st.spinner("Analyzing..."):
                tfidf_input = vectorizer.transform([job_text])
                pred = model.predict(tfidf_input)[0]
                proba = model.predict_proba(tfidf_input)[0]
                confidence = proba[pred] * 100

                if pred == 1:
                    st.error(f"⚠️ Verdict: FAKE ({confidence:.1f}% confidence)")
                else:
                    st.success(f"✅ Verdict: GENUINE ({confidence:.1f}% confidence)")

                st.subheader("Why this verdict? (LIME Explanation)")
                explainer = LimeTextExplainer(class_names=["Genuine", "Fake"])
                exp = explainer.explain_instance(job_text, predict_proba, num_features=10)

                for word, weight in exp.as_list():
                    if weight > 0:
                        st.markdown(f"🔴 **{word}** — pushes toward *Fake* ({weight:.3f})")
                    else:
                        st.markdown(f"🟢 **{word}** — pushes toward *Genuine* ({weight:.3f})")

# ======================================================
# TAB 2: Resume Match
# ======================================================
with tab2:
    st.title("🎯 Resume Match")
    st.write("Upload your resume (PDF) and paste a job description (or job URL) to see how well they match.")

    resume_file = st.file_uploader("Upload your resume (PDF):", type=["pdf"], key="resume_upload")

    resume_input = ""
    if resume_file is not None:
        reader = PyPDF2.PdfReader(resume_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                resume_input += page_text + " "
        st.success(f"Resume uploaded — extracted {len(resume_input.split())} words.")

    job_desc_input = st.text_area("Paste job description or URL:", height=180, key="job_desc_input")

    if st.button("Check Match", key="match_btn"):
        if not resume_input.strip() or not job_desc_input.strip():
            st.warning("Please upload your resume PDF and paste the job description/URL.")
        else:
            job_desc_text = job_desc_input.strip()

            if is_url(job_desc_text):
                with st.spinner("Scraping job description from URL..."):
                    scraped = scrape_job_text(job_desc_text)
                if scraped is None or len(scraped) < 50:
                    st.error("Could not scrape enough text from that URL. Try pasting the job description directly instead.")
                    st.stop()
                job_desc_text = scraped

            with st.spinner("Calculating match..."):
                match_vectorizer = TfidfVectorizer(stop_words="english")
                tfidf_matrix = match_vectorizer.fit_transform([resume_input, job_desc_text])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                match_percent = similarity * 100

                st.metric("Match Score", f"{match_percent:.1f}%")

                resume_words = set(match_vectorizer.build_analyzer()(resume_input))
                job_words = set(match_vectorizer.build_analyzer()(job_desc_text))

                matched = resume_words.intersection(job_words)
                missing = job_words - resume_words

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("✅ Matched Keywords")
                    st.markdown(" ".join([f"`{w}`" for w in sorted(matched)]) if matched else "No matched keywords found.")
                with col2:
                    st.subheader("⚠️ Missing Keywords")
                    st.markdown(" ".join([f"`{w}`" for w in sorted(missing)]) if missing else "No missing keywords — great match!")