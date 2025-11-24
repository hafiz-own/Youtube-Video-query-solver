import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
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

load_dotenv()  # Load Google API key


# =============================================
# Utility Functions
# =============================================


def extract_video_id(url: str):
    """Extract video ID from any YouTube URL."""
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url  # assume user gave plain ID


def load_transcript(video_id: str):
    """Download and return transcript text."""
    try:
        transcript_list = YouTubeTranscriptApi().fetch(
            video_id=video_id, languages=["en"]
        )
        transcript = " ".join(chunk.text for chunk in transcript_list)
        return transcript
    except TranscriptsDisabled:
        return None


def build_vectorstore(transcript_text: str):
    """Split text → embed → FAISS index."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])

    # embeddings = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2"
    # )
    embeddings = HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2", task="feature-extraction"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


def build_rag_chain(vector_store):
    """Create the final RAG pipeline."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    prompt = PromptTemplate(
        template="""
            You are a helpful assistant.
            Answer ONLY using the provided transcript context but do not mention transcript, mention video instead keeping abstraction. But you can go out of the video ONLY if needs to explain anything from within the video.
            If the context is insufficient, apologize nicely with the reason.

            Context:
            {context}

            Question: {question}
            """,
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    parser = StrOutputParser()

    rag_chain = parallel_chain | prompt | llm | parser
    return rag_chain


# =============================================
# Streamlit UI
# =============================================

st.set_page_config(page_title="YouTube RAG Chatbot", layout="centered")
st.title("YouTube RAG Chatbot")
st.write("Ask questions grounded in a video's transcript using Gemini + LangChain RAG.")

# Session state to store vector store and chain
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


# ------------------------------
# Step 1 — YouTube Link Input
# ------------------------------
with st.container(border=True):
    st.subheader("Enter YouTube Link")
    url = st.text_input(
        "Paste YouTube URL here:",
        placeholder="https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
    )

    if st.button("Process Video"):
        if not url.strip():
            st.warning("Please enter a valid YouTube link.")
        else:
            with st.spinner("Downloading transcript..."):
                video_id = extract_video_id(url)
                transcript = load_transcript(video_id)

            if not transcript:
                st.error("Transcript not available for this video.")
            else:
                with st.spinner("Indexing transcript (splitting + embeddings)..."):
                    vector_store = build_vectorstore(transcript)
                    rag_chain = build_rag_chain(vector_store)

                st.session_state.vector_store = vector_store
                st.session_state.rag_chain = rag_chain

                st.success("Video processed successfully! You can now ask questions.")


# ------------------------------
# Step 2 — Asking Questions
# ------------------------------
if st.session_state.rag_chain:
    with st.container(border=True):
        st.subheader("Ask a Question About the Video")

        question = st.text_input(
            "Your question:",
            placeholder="e.g. What is the video about?",
        )

        if st.button("Get Answer"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking..."):
                    answer = st.session_state.rag_chain.invoke(question)

                st.write("### Answer")
                st.write(answer)


st.markdown("---")
st.caption("Built with LangChain + FAISS + Gemini + Streamlit")
