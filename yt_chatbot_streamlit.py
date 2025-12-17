import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

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
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url


def load_transcript(video_id: str):
    try:
        transcript_list = YouTubeTranscriptApi().fetch(
            video_id=video_id, languages=["en"]
        )
        return " ".join(chunk.text for chunk in transcript_list)
    except TranscriptsDisabled:
        return None


def build_vectorstore(transcript_text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])

    embeddings = HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
    )

    return FAISS.from_documents(chunks, embeddings)


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
    /* ===== App Base ===== */
    body {
        background-color: #f9fafb;
        color: #0f172a;
    }

    .stApp {
        background-color: #f9fafb;
    }

    /* ===== Typography ===== */
    h1, h2, h3, h4 {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #0f172a;
        font-weight: 600;
        letter-spacing: -0.015em;
    }

    p, label {
        color: #475569;
        font-size: 0.95rem;
    }

    /* ===== Header Subtitle ===== */
    .subtitle {
        color: #475569;
        font-size: 1.05rem;
        margin-top: -6px;
    }

    /* ===== Cards ===== */
    .card {
        background-color: #ffffff;
        border-radius: 14px;
        padding: 28px;
        margin-bottom: 30px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    }

    /* ===== Inputs ===== */
    .stTextInput input {
        background-color: #ffffff;
        color: #0f172a;
        border-radius: 10px;
        padding: 12px 14px;
        border: 1px solid #d1d5db;
        font-size: 0.95rem;
    }

    .stTextInput input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
        outline: none;
    }

    .stTextInput input::placeholder {
        color: #9ca3af;
    }

    /* ===== Buttons (Modern SaaS Style) ===== */
    .stButton > button {
        background-color: #2563eb;
        color: #ffffff;
        border-radius: 10px;
        padding: 0.55em 1.4em;
        border: none;
        font-weight: 600;
        font-size: 0.9rem;
        transition: background-color 0.15s ease, transform 0.1s ease;
    }

    .stButton > button:hover {
        background-color: #1e40af;
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* ===== Answer Box ===== */
    .answer-box {
        background-color: #f1f5f9;
        border-radius: 12px;
        padding: 18px 20px;
        color: #0f172a;
        line-height: 1.7;
        margin-top: 14px;
        border: 1px solid #e5e7eb;
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
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 48px;
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
    <div style="text-align:center; margin-bottom: 40px;">
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


# =============================================
# Step 1 — Video Processing
# =============================================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Enter YouTube Video Link")

url = st.text_input(
    label="",
    placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
)

if st.button("Process Video"):
    if not url.strip():
        st.warning("Please enter a valid YouTube link.")
    else:
        with st.spinner("📥 Fetching transcript..."):
            transcript = load_transcript(extract_video_id(url))

        if not transcript:
            st.error("Transcript not available for this video.")
        else:
            with st.spinner("🧠 Building vector index..."):
                vector_store = build_vectorstore(transcript)
                st.session_state.rag_chain = build_rag_chain(vector_store)

            st.success("✅ Video processed! Ask your questions below.")

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
    )

    if st.button("✨ Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("🤔 Thinking..."):
                answer = st.session_state.rag_chain.invoke(question)

            st.markdown("#### 🧠 Answer")
            st.markdown(
                f"""
                <div class="answer-box">
                    {answer}
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
