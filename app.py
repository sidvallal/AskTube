import streamlit as st

from src.loaders.youtube_loader import get_transcript
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.chunker import split_text
from src.embeddings.embedding_model import load_embedding_model
from src.vectordb.chroma_manager import create_vector_store
from src.retriever.retriever import get_retriever
from src.llm.llm_loader import load_llm
from src.llm.qa_chain import build_rag_chain
from src.utils.helpers import extract_video_id

# ---------------------------
# Streamlit Config
# ---------------------------

st.set_page_config(
    page_title="AskTube AI",
    layout="centered"
)

# ---------------------------
# Cache Resources
# ---------------------------

@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource
def get_llm():
    return load_llm()


@st.cache_data(show_spinner=False)
def process_video(video_id: str):
    document = get_transcript(video_id)
    document.page_content = clean_text(document.page_content)
    chunks = split_text(document)
    return document, chunks


# ---------------------------
# Session State Defaults
# ---------------------------

if "status" not in st.session_state:
    st.session_state.status = "idle"   # idle | processing | ready | error

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "video_info" not in st.session_state:
    st.session_state.video_info = None  # dict: language, chunks, url

if "error_message" not in st.session_state:
    st.session_state.error_message = None


# ---------------------------
# Sidebar
# ---------------------------

with st.sidebar:
    st.header("📊 Video Status")

    status = st.session_state.status

    if status == "idle":
        st.info("No video processed yet.")

    elif status == "processing":
        st.warning(" Processing video...")

    elif status == "ready":
        st.success(" Ready for questions")

    elif status == "error":
        st.error(" Processing failed")
        if st.session_state.error_message:
            st.caption(st.session_state.error_message)

    st.divider()

    if st.session_state.video_info:
        info = st.session_state.video_info

        st.subheader("Video Info")

        st.image(
            f"https://img.youtube.com/vi/{st.session_state.video_id}/hqdefault.jpg",
            use_container_width=True
        )

        st.markdown(f"**Video ID:** `{st.session_state.video_id}`")
        st.markdown(f"**Language:** {info['language']}")
        st.markdown(f"**Chunks:** {info['chunks']}")
        st.markdown(f"🔗 [ Open on YouTube]({info['url']})")

        if st.button(" Reset", use_container_width=True):
            for key in ["status", "video_id", "video_info", "error_message", "rag_chain"]:
                st.session_state.pop(key, None)
            st.rerun()


# ---------------------------
# Title
# ---------------------------

st.title("🎥 AskTube AI")
st.write("Chat with any YouTube video using AI.")

# ---------------------------
# Video Processing
# ---------------------------

video_url = st.text_input(
    "YouTube Video URL",
    placeholder="e.g. https://www.youtube.com/watch?v=aircAruvnKk"
)

if st.button("Process Video", use_container_width=True):

    if not video_url:
        st.warning("Please enter a YouTube video URL.")
        st.stop()

    video_id = extract_video_id(video_url)

    st.session_state.status = "processing"
    st.session_state.error_message = None

    try:
        with st.spinner("Processing video..."):

            document, chunks = process_video(video_id)
            embedding_model = get_embedding_model()
            vector_store = create_vector_store(chunks, embedding_model)
            retriever = get_retriever(vector_store)
            llm = get_llm()
            rag_chain = build_rag_chain(llm, retriever)

            st.session_state.rag_chain = rag_chain
            st.session_state.video_id = video_id
            st.session_state.video_info = {
                "language": document.metadata.get("language", "Unknown"),
                "chunks": len(chunks),
                "url": document.metadata.get("source", video_url),
            }
            st.session_state.status = "ready"

        st.success("Video processed successfully!")
        st.info("You can now ask questions about this video.")
        st.rerun()

    except Exception as e:
        st.session_state.status = "error"
        st.session_state.error_message = str(e)
        st.error(str(e))


# ---------------------------
# Question Answering
# ---------------------------

if "rag_chain" in st.session_state:

    st.divider()
    st.subheader("Ask a Question")

    question = st.text_input(
        "Question",
        placeholder="What is this video about?"
    )

    if st.button("Get Answer", use_container_width=True):

        if not question:
            st.warning("Please enter a question.")
            st.stop()

        try:
            with st.spinner("Generating answer..."):
                response = st.session_state.rag_chain.invoke({"input": question})

            st.subheader("Answer")
            st.write(response["answer"])

        except Exception as e:
            st.error(str(e))