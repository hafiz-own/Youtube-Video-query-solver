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
# Theme State
# =============================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = st.get_option("theme.base") == "dark"

_, theme_col = st.columns([0.8, 0.2])
with theme_col:
    st.toggle("Dark mode", key="dark_mode")


# =============================================
# Custom CSS
# =============================================

if st.session_state.dark_mode:
    theme_vars = """
    :root {
        --bg-main: #0a1020;
        --bg-card: rgba(17, 26, 47, 0.84);
        --text-main: #e6edf8;
        --text-muted: #9fb1cc;
        --border-soft: rgba(148, 163, 184, 0.28);

        --brand: #7ab2ff;
        --brand-dark: #5b8dff;
        --ring: rgba(122, 178, 255, 0.34);
        --button-shadow: rgba(59, 130, 246, 0.32);

        --input-bg: rgba(11, 17, 31, 0.76);
        --input-placeholder: #8194b3;
        --answer-bg: rgba(10, 18, 34, 0.82);

        --chip-bg: rgba(59, 130, 246, 0.2);
        --chip-text: #cce1ff;
        --chip-border: rgba(96, 165, 250, 0.46);

        --hero-grad-1: rgba(59, 130, 246, 0.3);
        --hero-grad-2: rgba(56, 189, 248, 0.14);
        --glow-a: rgba(59, 130, 246, 0.24);
        --glow-b: rgba(14, 165, 233, 0.2);

        --alert-bg: rgba(15, 23, 42, 0.66);
        --card-shadow: 0 24px 42px rgba(2, 6, 23, 0.52);
    }
    """
else:
    theme_vars = """
    :root {
        --bg-main: #f3f7ff;
        --bg-card: rgba(255, 255, 255, 0.9);
        --text-main: #0f172a;
        --text-muted: #4b5f7a;
        --border-soft: #d8e2f2;

        --brand: #2f6cf6;
        --brand-dark: #2453ca;
        --ring: rgba(47, 108, 246, 0.2);
        --button-shadow: rgba(47, 108, 246, 0.28);

        --input-bg: rgba(255, 255, 255, 0.96);
        --input-placeholder: #8a98af;
        --answer-bg: #f8fbff;

        --chip-bg: #e9f1ff;
        --chip-text: #234484;
        --chip-border: #bfd5fb;

        --hero-grad-1: rgba(37, 99, 235, 0.18);
        --hero-grad-2: rgba(14, 165, 233, 0.1);
        --glow-a: rgba(37, 99, 235, 0.17);
        --glow-b: rgba(14, 165, 233, 0.15);

        --alert-bg: #f9fbff;
        --card-shadow: 0 20px 34px rgba(15, 23, 42, 0.1);
    }
    """

style_template = """
<style>
__THEME_VARS__

body {
    background-color: var(--bg-main);
    color: var(--text-main);
}

.stApp {
    background:
    radial-gradient(circle at 6% 2%, var(--glow-a), transparent 28%),
    radial-gradient(circle at 94% 4%, var(--glow-b), transparent 32%),
        var(--bg-main);
}

.block-container {
    max-width: 920px;
    padding-top: 1.7rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: var(--text-main);
    letter-spacing: -0.01em;
}

p, label {
    color: var(--text-muted);
}

.header-shell {
    background:
        linear-gradient(135deg, var(--hero-grad-1), var(--hero-grad-2)),
        var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 20px;
    padding: 24px 18px;
    margin-bottom: 22px;
    backdrop-filter: blur(8px);
    box-shadow: var(--card-shadow);
}

.subtitle {
    color: var(--text-muted);
    font-size: 1.02rem;
    margin-top: -4px;
}

.hero-badge {
    display: inline-block;
    margin-bottom: 10px;
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--chip-text);
    background: var(--chip-bg);
    border: 1px solid var(--chip-border);
    border-radius: 999px;
    padding: 5px 11px;
}

.card {
    background: var(--bg-card);
    border-radius: 18px;
    padding: 26px;
    margin-bottom: 24px;
    border: 1px solid var(--border-soft);
    box-shadow: var(--card-shadow);
    backdrop-filter: blur(8px);
}

.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 10px;
    color: var(--chip-text);
    background: var(--chip-bg);
    border: 1px solid var(--chip-border);
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
}

.stTextInput input {
    background-color: var(--input-bg);
    color: var(--text-main);
    border-radius: 12px;
    padding: 12px 14px;
    border: 1px solid var(--border-soft);
    font-size: 0.96rem;
}

.stTextInput input:focus {
    border-color: var(--brand);
    box-shadow: 0 0 0 4px var(--ring);
    outline: none;
}

[data-baseweb="input"] {
    background: transparent !important;
}

[data-baseweb="base-input"] {
    background-color: var(--input-bg) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 12px !important;
}

.stTextInput input::placeholder {
    color: var(--input-placeholder);
}

.stButton > button {
    width: 100%;
    background: linear-gradient(118deg, var(--brand), var(--brand-dark));
    color: #ffffff;
    border-radius: 11px;
    padding: 0.62em 1.4em;
    border: 0;
    font-weight: 700;
    font-size: 0.92rem;
    transition: transform 0.12s ease, box-shadow 0.16s ease, filter 0.16s ease;
    box-shadow: 0 10px 20px var(--button-shadow);
}

.stButton > button:hover {
    transform: translateY(-1px);
    filter: brightness(1.03);
    box-shadow: 0 13px 24px var(--button-shadow);
}

.stButton > button:active {
    transform: translateY(0px);
}

.answer-box {
    background: var(--answer-bg);
    border-radius: 14px;
    padding: 18px 20px;
    color: var(--text-main);
    line-height: 1.72;
    margin-top: 14px;
    border: 1px solid var(--border-soft);
}

.stAlert {
    background: var(--alert-bg) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-soft) !important;
}

[data-testid="stToggle"] label p {
    color: var(--text-muted);
    font-size: 0.9rem;
}

.footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.84rem;
    margin-top: 36px;
}

@media (max-width: 768px) {
    .card {
        padding: 20px;
    }

    .header-shell {
        padding: 20px 14px;
    }
}
</style>
"""

st.markdown(
    style_template.replace("__THEME_VARS__", theme_vars),
    unsafe_allow_html=True,
)


# =============================================
# Header
# =============================================

st.markdown(
    """
    <div class="header-shell" style="text-align:center;">
        <span class="hero-badge">Video AI Assistant</span>
        <h1>🎥 YouTube RAG Chatbot</h1>
        <p class="subtitle">
            Ask intelligent questions grounded directly in a YouTube video transcript
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
    label="YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
    label_visibility="collapsed",
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
        label="Question",
        placeholder="What is the main idea of the video?",
        label_visibility="collapsed",
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
