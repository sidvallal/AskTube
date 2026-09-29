import os
import streamlit as st
import traceback

from dotenv import load_dotenv
from langsmith import traceable

from src.loaders.youtube_loader import get_transcript
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.chunker import split_text
from src.embeddings.embedding_model import load_embedding_model
from src.vectordb.chroma_manager import create_vector_store
from src.retriever.retriever import get_retriever
from src.llm.llm_loader import load_llm
from src.llm.qa_chain import build_rag_chain
from src.utils.helpers import (
    extract_video_id,
    get_video_title,
    truncate_text,
)
from src.utils.config import Config


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"

# Configure LangSmith API key if available
if Config.LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = Config.LANGSMITH_API_KEY

# Configure LangSmith project
os.environ["LANGSMITH_PROJECT"] = Config.LANGSMITH_PROJECT


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AskTube AI",
    layout="centered"
)


SUGGESTED_QUESTIONS = [
    "Summarize this video",
    "What are the key takeaways?",
    "Explain this like I'm five",
]


# ============================================================
# CACHE RESOURCES
# ============================================================

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Load the embedding model only once."""
    return load_embedding_model()


@st.cache_resource(show_spinner=False)
def get_llm():
    """Load the LLM only once."""
    return load_llm()


@st.cache_data(show_spinner=False)
def cached_get_transcript(video_id: str):
    """Fetch transcript. Cached per video ID."""
    return get_transcript(video_id)


@st.cache_data(show_spinner=False)
def cached_get_video_title(video_id: str):
    """Fetch the human-readable video title. Cached per video ID."""
    return get_video_title(video_id)


# ============================================================
# SESSION STATE DEFAULTS
# ============================================================

if "status" not in st.session_state:
    st.session_state.status = "idle"

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "video_info" not in st.session_state:
    st.session_state.video_info = None

if "error_message" not in st.session_state:
    st.session_state.error_message = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# LANGSMITH TRACING
# ============================================================

import traceback
from langsmith import traceable


@traceable(
    name="AskTube Question Handler",
    tags=["AskTube", "Question-Answering"],
    metadata={"project": "AskTube-AI"}
)
def handle_question(question: str):
    """
    Runs a question through the RAG chain,
    stores the answer and source snippets.
    """

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    try:
        # Invoke RAG chain
        response = st.session_state.rag_chain.invoke(
            {"input": question}
        )

        answer = response["answer"]

        # Extract source snippets
        source_snippets = []

        for doc in response.get("context", []):
            source_snippets.append(
                truncate_text(doc.page_content, 250)
            )

        # Store assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": source_snippets,
        })

    except Exception as e:

        # Print the complete original error
        print("\n========== ASK TUBE ERROR ==========")
        traceback.print_exc()
        print("====================================\n")

        # Store the error message
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error: {str(e)}",
            "sources": [],
        })
# ============================================================
# SIDEBAR: CONTROLS + STATUS + VIDEO INFO
# ============================================================

with st.sidebar:

    st.header(":material/smart_display: AskTube AI")

    video_url = st.text_input(
        "YouTube Video URL",
        placeholder="e.g. https://www.youtube.com/watch?v=aircAruvnKk"
    )

    process_clicked = st.button(
        ":material/play_circle: Process Video",
        width="stretch"
    )

    st.divider()

    st.subheader(":material/monitoring: Video Status")

    status = st.session_state.status

    if status == "idle":
        st.info(":material/info: No video processed yet.")

    elif status == "ready":
        st.success(":material/check_circle: Ready for questions")

    elif status == "error":
        st.error(":material/error: Processing failed")

        if st.session_state.error_message:
            st.caption(st.session_state.error_message)

    # --------------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------------

    if st.session_state.video_info:

        info = st.session_state.video_info

        st.divider()
        st.subheader(":material/movie: Video Info")

        st.image(
            f"https://img.youtube.com/vi/{st.session_state.video_id}/hqdefault.jpg",
            width="stretch"
        )

        st.markdown(f"**{info['title']}**")
        st.markdown(f"**Language:** {info['language']}")

        st.markdown(
            f":material/open_in_new: [Open on YouTube]({info['url']})"
        )

        # ----------------------------------------------------
        # RESET BUTTON
        # ----------------------------------------------------

        if st.button(
            ":material/refresh: Reset",
            width="stretch"
        ):

            for key in [
                "status",
                "video_id",
                "video_info",
                "error_message",
                "rag_chain",
                "messages",
            ]:
                st.session_state.pop(key, None)

            st.rerun()


# ============================================================
# VIDEO PROCESSING
# ============================================================

if process_clicked:

    if not video_url:
        st.toast(
            ":material/warning: Please enter a YouTube video URL."
        )
        st.stop()

    video_id = extract_video_id(video_url)

    st.session_state.status = "processing"
    st.session_state.error_message = None

    try:

        with st.spinner("Processing video..."):

            # -----------------------------------------------
            # STEP 1: LOAD YOUTUBE TRANSCRIPT
            # -----------------------------------------------

            document = cached_get_transcript(video_id)

            # -----------------------------------------------
            # STEP 2: CLEAN TRANSCRIPT
            # -----------------------------------------------

            document.page_content = clean_text(
                document.page_content
            )

            # -----------------------------------------------
            # STEP 3: SPLIT TRANSCRIPT INTO CHUNKS
            # -----------------------------------------------

            chunks = split_text(document)

            # -----------------------------------------------
            # STEP 4: GET VIDEO TITLE
            # -----------------------------------------------

            title = cached_get_video_title(video_id)

            # -----------------------------------------------
            # STEP 5: LOAD EMBEDDING MODEL
            # -----------------------------------------------

            embedding_model = get_embedding_model()

            # -----------------------------------------------
            # STEP 6: CREATE VECTOR STORE
            # -----------------------------------------------

            vector_store = create_vector_store(
                chunks,
                embedding_model
            )

            # -----------------------------------------------
            # STEP 7: CREATE RETRIEVER
            # -----------------------------------------------

            retriever = get_retriever(vector_store)

            # -----------------------------------------------
            # STEP 8: LOAD LLM
            # -----------------------------------------------

            llm = get_llm()

            # -----------------------------------------------
            # STEP 9: BUILD RAG CHAIN
            # -----------------------------------------------

            rag_chain = build_rag_chain(
                llm,
                retriever
            )

            # -----------------------------------------------
            # STEP 10: SAVE IN SESSION STATE
            # -----------------------------------------------

            st.session_state.rag_chain = rag_chain

            st.session_state.video_id = video_id

            st.session_state.video_info = {
                "title": title,
                "language": document.metadata.get(
                    "language", "Unknown"
                ),
                "chunks": len(chunks),
                "url": document.metadata.get(
                    "source", video_url
                ),
            }

            st.session_state.status = "ready"

            # Reset messages for new video
            st.session_state.messages = []

        st.toast(
            ":material/check_circle: Video processed successfully!"
        )

        st.rerun()

    except Exception as e:

        st.session_state.status = "error"
        st.session_state.error_message = str(e)

        st.toast(
            f":material/error: Processing failed: {e}"
        )

        st.rerun()


# ============================================================
# MAIN TITLE
# ============================================================

st.title("AskTube AI", anchor=False)

st.caption(
    ":material/smart_display: Chat with any YouTube video using AI."
)


# ============================================================
# CHAT INTERFACE
# ============================================================

if "rag_chain" in st.session_state:

    st.divider()

    if st.session_state.video_info:

        st.subheader(
            f":material/forum: {st.session_state.video_info['title']}"
        )

    else:

        st.subheader(
            ":material/forum: Chat with the Video"
        )

    # --------------------------------------------------------
    # SUGGESTED QUESTIONS
    # --------------------------------------------------------

    if not st.session_state.messages:

        st.caption("Try one of these to get started:")

        cols = st.columns(len(SUGGESTED_QUESTIONS))

        for col, suggestion in zip(
            cols,
            SUGGESTED_QUESTIONS
        ):

            with col:

                if st.button(
                    suggestion,
                    width="stretch"
                ):

                    with st.spinner("Thinking..."):
                        handle_question(suggestion)

                    st.rerun()

    # --------------------------------------------------------
    # RENDER CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.messages:

        avatar = (
            ":material/person:"
            if message["role"] == "user"
            else ":material/smart_toy:"
        )

        with st.chat_message(
            message["role"],
            avatar=avatar
        ):

            st.write(message["content"])

            # Display source snippets
            if message["role"] == "assistant":

                sources = message.get("sources", [])

                if sources:

                    with st.expander("View Sources"):

                        for index, source in enumerate(
                            sources,
                            start=1
                        ):

                            st.markdown(
                                f"**Source {index}:**"
                            )

                            st.write(source)

    # --------------------------------------------------------
    # NEW USER INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask something about this video..."
    )

    if question:

        with st.spinner("Thinking..."):
            handle_question(question)

        st.rerun()

else:

    st.info(
        ":material/arrow_back: Enter a YouTube URL in the sidebar "
        "and click **Process Video** to get started."
    )