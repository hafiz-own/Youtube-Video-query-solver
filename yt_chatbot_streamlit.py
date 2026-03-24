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

_, theme_col = st.columns([0.84, 0.16])
with theme_col:
    st.toggle("🌙 Dark", key="dark_mode")


# =============================================
# Custom CSS
# =============================================

if st.session_state.dark_mode:
    theme_vars = """
    :root {
        --bg-main: #0b1220;
        --bg-card: rgba(15, 23, 42, 0.82);
        --text-main: #e5edf7;
        --text-muted: #98a9c2;
        --border-soft: rgba(148, 163, 184, 0.22);
        --brand: #60a5fa;
        --brand-dark: #3b82f6;
        --ring: rgba(96, 165, 250, 0.28);
        --answer-bg: rgba(30, 41, 59, 0.9);
        --chip-bg: rgba(37, 99, 235, 0.16);
        --chip-text: #bfdbfe;
        --chip-border: rgba(96, 165, 250, 0.45);
        --card-shadow: 0 20px 34px rgba(2, 6, 23, 0.35);
    }
    """
else:
    theme_vars = """
    :root {
        --bg-main: #f4f7ff;
        --bg-card: rgba(255, 255, 255, 0.88);
        --text-main: #0f172a;
        --text-muted: #475569;
        --border-soft: #dbe5f1;
        --brand: #2563eb;
        --brand-dark: #1d4ed8;
        --ring: rgba(37, 99, 235, 0.18);
        --answer-bg: #f8fbff;
        --chip-bg: #e9f1ff;
        --chip-text: #1e3a8a;
        --chip-border: #bdd6ff;
        --card-shadow: 0 18px 34px rgba(15, 23, 42, 0.08);
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
        radial-gradient(circle at 6% 2%, rgba(59, 130, 246, 0.17), transparent 26%),
        radial-gradient(circle at 94% 4%, rgba(14, 165, 233, 0.17), transparent 30%),
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
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.16), rgba(14, 165, 233, 0.08));
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
    background-color: rgba(255, 255, 255, 0.85);
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
    box-shadow: 0 10px 20px rgba(37, 99, 235, 0.26);
}

.stButton > button:hover {
    transform: translateY(-1px);
    filter: brightness(1.03);
    box-shadow: 0 13px 24px rgba(37, 99, 235, 0.3);
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
    border-radius: 12px !important;
    border: 1px solid var(--border-soft) !important;
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
