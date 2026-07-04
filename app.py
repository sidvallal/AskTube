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
    """Load the embedding model only once."""
    return load_embedding_model()


@st.cache_resource
def get_llm():
    """Load the LLM only once."""
    return load_llm()


@st.cache_data(show_spinner=False)
def process_video(video_id: str):
    """
    Fetch transcript, clean it and split it into chunks.
    This is cached for each unique video ID.
    """
    document = get_transcript(video_id)
    document.page_content = clean_text(document.page_content)
    chunks = split_text(document)

    return document, chunks


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

    try:
        with st.spinner("Processing video..."):
            document, chunks = process_video(video_id)
            ...

            # Cached embedding model
            embedding_model = get_embedding_model()

            # Create vector database
            vector_store = create_vector_store(
                chunks,
                embedding_model
            )

            # Retriever
            retriever = get_retriever(vector_store)

            # Cached LLM
            llm = get_llm()

            # Build RAG chain
            rag_chain = build_rag_chain(
                llm,
                retriever
            )

            st.session_state.rag_chain = rag_chain

        st.success("Video processed successfully!")

        col1, col2 = st.columns(2)

        col1.metric(
            "Language",
            document.metadata.get("language", "Unknown")
        )

        col2.metric(
            "Chunks",
            len(chunks)
        )

        st.info("You can now ask questions about this video.")

    except Exception as e:
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

                response = st.session_state.rag_chain.invoke(
                    {
                        "input": question
                    }
                )

            st.subheader("Answer")

            st.write(response["answer"])

        except Exception as e:
            st.error(str(e))