import streamlit as st

from src.loaders.youtube_loader import get_transcript
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.chunker import split_text
from src.embeddings.embedding_model import load_embedding_model
from src.vectordb.chroma_manager import create_vector_store
from src.retriever.retriever import get_retriever
from src.llm.llm_loader import load_llm
from src.llm.qa_chain import build_rag_chain
from src.utils.helpers import extract_video_id, get_video_title, truncate_text

# ---------------------------
# Streamlit Config
# ---------------------------

st.set_page_config(
    page_title="AskTube AI",
    layout="centered"
)

SUGGESTED_QUESTIONS = [
    "Summarize this video",
    "What are the key takeaways?",
    "Explain this like I'm five",
]

# ---------------------------
# Cache Resources
# ---------------------------

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


# ---------------------------
# Session State Defaults
# ---------------------------

if "status" not in st.session_state:
    st.session_state.status = "idle"   # idle | processing | ready | error

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "video_info" not in st.session_state:
    st.session_state.video_info = None  # dict: title, language, chunks, url

if "error_message" not in st.session_state:
    st.session_state.error_message = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------
# Helper: handle a question end-to-end
# ---------------------------

def handle_question(question: str):
    """
    Runs a question through the RAG chain, stores the user question
    and assistant answer (with source snippets) in session state.
    """

    st.session_state.messages.append({"role": "user", "content": question})

    try:
        response = st.session_state.rag_chain.invoke({"input": question})
        answer = response["answer"]

        source_snippets = []
        for doc in response.get("context", []):
            source_snippets.append(truncate_text(doc.page_content, 250))

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": source_snippets,
        })

    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f":material/error: Error: {e}",
            "sources": [],
        })


# ---------------------------
# Sidebar: Controls + Status + Video Info
# ---------------------------

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
        # st.markdown(f"**Chunks:** {info['chunks']}")
        st.markdown(f":material/open_in_new: [Open on YouTube]({info['url']})")

        if st.button(":material/refresh: Reset", width="stretch"):
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


# ---------------------------
# Video Processing (triggered from sidebar button)
# ---------------------------

if process_clicked:

    if not video_url:
        st.toast(":material/warning: Please enter a YouTube video URL.")
        st.stop()

    video_id = extract_video_id(video_url)
    st.session_state.status = "processing"
    st.session_state.error_message = None

    try:
        with st.spinner("Processing video..."):

            document = cached_get_transcript(video_id)
            document.page_content = clean_text(document.page_content)
            chunks = split_text(document)
            title = cached_get_video_title(video_id)

            embedding_model = get_embedding_model()
            vector_store = create_vector_store(chunks, embedding_model)
            retriever = get_retriever(vector_store)

            llm = get_llm()
            rag_chain = build_rag_chain(llm, retriever)

            st.session_state.rag_chain = rag_chain
            st.session_state.video_id = video_id
            st.session_state.video_info = {
                "title": title,
                "language": document.metadata.get("language", "Unknown"),
                "chunks": len(chunks),
                "url": document.metadata.get("source", video_url),
            }
            st.session_state.status = "ready"
            st.session_state.messages = []  # reset chat for new video

        st.toast(":material/check_circle: Video processed successfully!")
        st.rerun()

    except Exception as e:
        st.session_state.status = "error"
        st.session_state.error_message = str(e)
        st.toast(f":material/error: Processing failed: {e}")
        st.rerun()


# ---------------------------
# Title
# ---------------------------

st.title("AskTube AI", anchor=False)
st.caption(":material/smart_display: Chat with any YouTube video using AI.")

# ---------------------------
# Chat Interface
# ---------------------------

if "rag_chain" in st.session_state:

    st.divider()

    if st.session_state.video_info:
        st.subheader(f":material/forum: {st.session_state.video_info['title']}")
    else:
        st.subheader(":material/forum: Chat with the Video")

    # Suggested question chips (only before the first question)
    if not st.session_state.messages:
        st.caption("Try one of these to get started:")
        cols = st.columns(len(SUGGESTED_QUESTIONS))
        for col, suggestion in zip(cols, SUGGESTED_QUESTIONS):
            with col:
                if st.button(suggestion, width="stretch"):
                    with st.spinner("Thinking..."):
                        handle_question(suggestion)
                    st.rerun()

    # Render chat history
    for message in st.session_state.messages:
        avatar = ":material/person:" if message["role"] == "user" else ":material/smart_toy:"
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])

    # New user input
    question = st.chat_input("Ask something about this video...")

    if question:
        with st.spinner("Thinking..."):
            handle_question(question)
        st.rerun()

else:
    st.info(":material/arrow_back: Enter a YouTube URL in the sidebar and click **Process Video** to get started.")