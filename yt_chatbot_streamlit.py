import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from urllib.parse import parse_qs, urlparse

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
    )


# How many prior turns (user + assistant messages) to feed back in as
# conversational memory. Kept small so the model doesn't lose focus on
# the transcript itself.
HISTORY_WINDOW = 6


def condense_question(llm, history, question):
    """Rewrite a follow-up question into a standalone one using recent history."""
    if not history:
        return question

    turns = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-HISTORY_WINDOW:]
    )
    prompt = (
        "Rewrite the follow-up question below as a standalone question that "
        "includes any context it implicitly relies on from the chat history. "
        "Output only the rewritten question, nothing else.\n\n"
        f"Chat history:\n{turns}\n\n"
        f"Follow-up question: {question}\n"
        "Standalone question:"
    )
    return llm.invoke(prompt).content.strip()


def generate_answer(llm, retriever, history, question):
    """Answer a question using retrieved transcript context plus chat memory."""
    standalone_question = condense_question(llm, history, question)
    docs = retriever.invoke(standalone_question)
    context = "\n\n".join(doc.page_content for doc in docs)

    system = SystemMessage(
        content=(
            "You are a helpful assistant answering questions about a YouTube "
            "video using only the transcript context below. If the context "
            "doesn't contain the answer, say so politely instead of guessing.\n\n"
            f"Context:\n{context}"
        )
    )

    memory_messages = [
        (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
        for m in history[-HISTORY_WINDOW:]
    ]

    messages = [system, *memory_messages, HumanMessage(content=question)]
    return llm.invoke(messages).content


# =============================================
# Page Config
# =============================================

st.set_page_config(
    page_title="YouTube AI Chat",
    page_icon="🎥",
    layout="centered",
)


# =============================================
# Theme-safe CSS
#
# No hand-rolled light/dark palettes here — everything below reads from
# Streamlit's own theme variables (--background-color, --text-color, etc.),
# which update automatically when the user switches themes from the
# Settings menu. That's what keeps this in sync with light/dark mode
# instead of fighting it.
# =============================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 760px;
        padding-top: 2.2rem;
        padding-bottom: 6rem;
    }

    h1 {
        font-size: 1.6rem;
        margin-bottom: 0.1rem;
    }

    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 10px;
    }

    section[data-testid="stSidebar"] .stButton > button {
        font-weight: 600;
    }

    [data-testid="stChatMessageContent"] {
        line-height: 1.65;
    }

    .stChatInput textarea {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================
# Session State
# =============================================

st.session_state.setdefault("messages", [])
st.session_state.setdefault("retriever", None)
st.session_state.setdefault("video_id", "")


# =============================================
# Sidebar — video loading
# =============================================

with st.sidebar:
    st.subheader("Video")

    video_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
        key="video_url_input",
    )

    process_clicked = st.button("Process video", use_container_width=True)

    if process_clicked:
        if not video_url.strip():
            st.warning("Enter a YouTube link first.")
        else:
            new_video_id = extract_video_id(video_url)
            if not new_video_id:
                st.error("Couldn't read a video ID from that link.")
            elif (
                new_video_id == st.session_state.video_id
                and st.session_state.retriever is not None
            ):
                st.info("Already loaded — continue chatting below.")
            else:
                with st.spinner("Fetching transcript..."):
                    transcript = load_transcript(new_video_id)

                if not transcript:
                    st.error("No transcript available for this video.")
                else:
                    with st.spinner("Indexing transcript..."):
                        vector_store = build_vectorstore(transcript)

                    st.session_state.retriever = vector_store.as_retriever(
                        search_kwargs={"k": 4}
                    )
                    st.session_state.video_id = new_video_id
                    st.session_state.messages = []
                    st.success("Video ready — ask away below.")

    if st.session_state.video_id:
        st.caption(f"Loaded video: `{st.session_state.video_id}`")
        if st.session_state.messages and st.button(
            "Clear chat", use_container_width=True
        ):
            st.session_state.messages = []
            st.rerun()


# =============================================
# Main — chat
# =============================================

st.title("🎥 YouTube RAG Chatbot")
st.caption("Ask questions grounded in a YouTube video's transcript.")

if not st.session_state.retriever:
    st.info(
        "Add a YouTube link in the sidebar and click **Process video** to get started."
    )
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = st.chat_input(
    "Ask something about the video..."
    if st.session_state.retriever
    else "Process a video in the sidebar first...",
    disabled=not st.session_state.retriever,
)

if prompt:
    prior_history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate_answer(
                get_llm(), st.session_state.retriever, prior_history, prompt
            )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})