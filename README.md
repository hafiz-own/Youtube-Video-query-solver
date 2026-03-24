# YouTube Video Query Solver

Ask questions about any YouTube video and get answers grounded in its transcript using RAG (LangChain + FAISS + Gemini + Streamlit).

## Live Demo

[https://youtube-video-query-solver-jyshbtm52x298tjgarzx2u.streamlit.app/
](https://youtube-video-query-solver-jyshbtm52x298tjgarzx2u.streamlit.app/)

## Features

- Extracts video transcript and builds a semantic vector index
- Answers only from retrieved video context
- Fast repeated runs via caching for transcript and embeddings
- Modern Streamlit UI with light and dark mode toggle

## Tech Stack

- Streamlit (UI)
- LangChain (RAG orchestration)
- FAISS (vector store)
- Gemini (`gemini-2.5-flash`) for generation
- Hugging Face endpoint embeddings (`all-MiniLM-L6-v2`)

## Requirements

- Python 3.12 (recommended)
- Google Gemini API key
- Hugging Face token

## Why Python 3.12

`streamlit` depends on `pyarrow`. On Python 3.14, `pyarrow` may attempt a source build and fail without system tools (for example `cmake`). Python 3.12 uses prebuilt wheels and installs cleanly.

## Setup

### 1) Create virtual environment

```bash
~/.pyenv/versions/3.12.12/bin/python -m venv .venv
. .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3) Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

## Run Locally

```bash
.venv/bin/streamlit run yt_chatbot_streamlit.py
```

Open the URL printed by Streamlit (usually http://localhost:8501).

## How to Use

1. Paste a YouTube URL.
2. Click **Process Video**.
3. Ask questions in the second section.
4. Toggle dark mode from the top-right control.

## Project Structure

- `yt_chatbot_streamlit.py` — app code and UI
- `requirements.txt` — Python dependencies
- `.gitignore` — ignored local/runtime files

## Troubleshooting

- **Import errors in editor**: Select `.venv/bin/python` as the interpreter and reload VS Code.
- **`pyarrow` install failure**: Use Python 3.12, then recreate `.venv`.
- **No transcript found**: Some videos do not provide English transcripts.
- **Empty answers**: Ensure API keys are set and valid in `.env`.
