import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from urllib.parse import parse_qs, urlparse

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()


# =============================================
# Utility Functions
# =============================================


def extract_video_id(url: str):
    cleaned_url = url.strip()
    parsed = urlparse(cleaned_url)

    if not parsed.netloc:
        return cleaned_url

    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")

    query_video_id = parse_qs(parsed.query).get("v", [""])[0]
    if query_video_id:
        return query_video_id

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) >= 2 and path_parts[0] in {"embed", "shorts", "live"}:
        return path_parts[1]

    return ""


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_transcript(video_id: str):
    try:
        transcript_list = YouTubeTranscriptApi().fetch(
            video_id=video_id, languages=["en"]
        )
        return " ".join(chunk.text for chunk in transcript_list)
    except TranscriptsDisabled:
        return None


@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    return HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
    )


@st.cache_resource(show_spinner=False)
def build_vectorstore(transcript_text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])

    return FAISS.from_documents(chunks, get_embeddings_model())


def build_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
    )

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY using the provided video context.
        If context is insufficient, apologize politely.

        Context:
        {context}

        Question: {question}
        """,
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# =============================================
# Page Config
# =============================================

st.set_page_config(
    page_title="YouTube AI Chat",
    page_icon="🎥",
    layout="centered",
)


# =============================================
# Custom CSS
# =============================================

st.markdown(
    """
    <style>
    :root {
        --bg-main: #f6f8fc;
        --bg-card: #ffffff;
        --text-main: #0f172a;
        --text-muted: #475569;
        --border-soft: #e2e8f0;
        --brand: #2563eb;
        --brand-dark: #1d4ed8;
    }

    /* ===== App Base ===== */
    body {
        background-color: var(--bg-main);
        color: var(--text-main);
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(37, 99, 235, 0.08), transparent 42%),
            radial-gradient(circle at top left, rgba(14, 165, 233, 0.08), transparent 36%),
            var(--bg-main);
    }

    .block-container {
        max-width: 860px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }

    /* ===== Typography ===== */
    h1, h2, h3, h4 {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--text-main);
        font-weight: 600;
        letter-spacing: -0.015em;
    }

    p, label {
        color: var(--text-muted);
        font-size: 0.95rem;
    }

    /* ===== Header Subtitle ===== */
    .subtitle {
        color: var(--text-muted);
        font-size: 1.05rem;
        margin-top: -6px;
    }

    .header-shell {
        background-color: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(37, 99, 235, 0.15);
        border-radius: 18px;
        padding: 20px 16px;
        backdrop-filter: blur(6px);
    }

    /* ===== Cards ===== */
    .card {
        background-color: var(--bg-card);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 30px;
        border: 1px solid var(--border-soft);
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }

    .status-chip {
        display: inline-block;
        margin-top: 8px;
        background-color: #eff6ff;
        color: #1e3a8a;
        border: 1px solid #bfdbfe;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* ===== Inputs ===== */
    .stTextInput input {
        background-color: #ffffff;
        color: var(--text-main);
        border-radius: 10px;
        padding: 12px 14px;
        border: 1px solid #cbd5e1;
        font-size: 0.95rem;
    }

    .stTextInput input:focus {
        border-color: var(--brand);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
        outline: none;
    }

    .stTextInput input::placeholder {
        color: #9ca3af;
    }

    /* ===== Buttons (Modern SaaS Style) ===== */
    .stButton > button {
        width: 100%;
        background: linear-gradient(120deg, var(--brand), var(--brand-dark));
        color: #ffffff;
        border-radius: 10px;
        padding: 0.6em 1.4em;
        border: none;
        font-weight: 600;
        font-size: 0.9rem;
        transition: transform 0.1s ease, box-shadow 0.15s ease;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.24);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.28);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* ===== Answer Box ===== */
    .answer-box {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 18px 20px;
        color: var(--text-main);
        line-height: 1.7;
        margin-top: 14px;
        border: 1px solid #dbeafe;
    }

    /* ===== Alerts ===== */
    .stAlert-success {
        background-color: #ecfdf5;
        color: #065f46;
        border-radius: 10px;
    }

    .stAlert-warning {
        background-color: #fffbeb;
        color: #92400e;
        border-radius: 10px;
    }

    .stAlert-error {
        background-color: #fef2f2;
        color: #991b1b;
        border-radius: 10px;
    }

    /* ===== Footer ===== */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================
# Header
# =============================================

st.markdown(
    """
    <div class="header-shell" style="text-align:center; margin-bottom: 30px;">
        <h1>🎥 YouTube RAG Chatbot</h1>
        <p class="subtitle">
            Ask intelligent questions grounded directly in a YouTube video
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================
# Session State
# =============================================

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "active_video_id" not in st.session_state:
    st.session_state.active_video_id = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""


# =============================================
# Step 1 — Video Processing
# =============================================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Enter YouTube Video Link")

url = st.text_input(
    label="",
    placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
    key="video_url_input",
)

video_id = extract_video_id(url)

if st.button("Process Video", key="process_video_btn"):
    if not url.strip():
        st.warning("Please enter a valid YouTube link.")
    elif not video_id:
        st.error("Could not extract a valid YouTube video ID.")
    elif st.session_state.active_video_id == video_id and st.session_state.rag_chain:
        st.success("Video already processed. Ask your questions below.")
    else:
        with st.spinner("📥 Fetching transcript..."):
            transcript = load_transcript(video_id)

        if not transcript:
            st.error("Transcript not available for this video.")
        else:
            with st.spinner("🧠 Building vector index..."):
                vector_store = build_vectorstore(transcript)
                st.session_state.rag_chain = build_rag_chain(vector_store)
                st.session_state.active_video_id = video_id
                st.session_state.last_answer = ""

            st.success("✅ Video processed! Ask your questions below.")

if st.session_state.active_video_id:
    st.markdown(
        f"""
        <div class="status-chip">Loaded video ID: {st.session_state.active_video_id}</div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)


# =============================================
# Step 2 — Q&A
# =============================================

if st.session_state.rag_chain:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💬 Step 2 — Ask a Question")

    question = st.text_input(
        label="",
        placeholder="What is the main idea of the video?",
        key="question_input",
    )

    if st.button("✨ Get Answer", key="get_answer_btn"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("🤔 Thinking..."):
                st.session_state.last_answer = st.session_state.rag_chain.invoke(question)

    if st.session_state.last_answer:
        st.markdown("#### 🧠 Answer")
        st.markdown(
            f"""
            <div class="answer-box">
                {st.session_state.last_answer}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================
# Footer
# =============================================

st.markdown(
    """
    <div class="footer">
        Built with ❤️ using LangChain · FAISS · Gemini · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
