import streamlit as st

from src.loaders.youtube_loader import get_transcript
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.chunker import split_text
from src.embeddings.embedding_model import load_embedding_model
from src.vectordb.chroma_manager import create_vector_store
from src.retriever.retriever import get_retriever, retrieve_documents

st.set_page_config(
    page_title="AskTube AI",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 AskTube AI")
st.markdown("Enter a YouTube Video ID to process and analyze the video's content.")

video_id = st.text_input(
    "YouTube Video ID",
    placeholder="e.g. aircAruvnKk"
)

if st.button("Process Video", use_container_width=True):

    if not video_id:
        st.warning("Please enter a YouTube Video ID.")
        st.stop()

    try:
        with st.spinner("Processing video..."):

            # Fetch transcript
            document = get_transcript(video_id)

            # Clean transcript
            document.page_content = clean_text(document.page_content)

            # Split transcript into chunks
            chunks = split_text(document)

            # Load embedding model
            embedding_model = load_embedding_model()

            # Create vector store
            vector_store = create_vector_store(
                chunks=chunks,
                embedding_model=embedding_model
            )

            # Create retriever
            retriever = get_retriever(vector_store)

            # Store retriever in session state
            st.session_state.retriever = retriever

            # Generate sample embedding
            sample_embedding = embedding_model.embed_query(
                chunks[0].page_content
            )

        st.success("Video processed and indexed successfully!")

        # Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Language",
                document.metadata.get("language", "Unknown")
            )

        with col2:
            st.metric(
                "Chunks Created",
                len(chunks)
            )

        with col3:
            st.metric(
                "Embedding Dimension",
                len(sample_embedding)
            )

        st.info(
            "Transcript has been stored in ChromaDB and is ready for question answering."
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")


# ---------------------------
# Question Section
# ---------------------------

if "retriever" in st.session_state:

    st.divider()
    st.subheader("Ask Questions About the Video")

    query = st.text_input(
        "Ask a question",
        placeholder="What is this video about?"
    )

    if st.button("Search Answer Context"):

        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching relevant content..."):

                docs = retrieve_documents(
                    st.session_state.retriever,
                    query
                )

            st.success(f"Retrieved {len(docs)} relevant chunks.")

            with st.expander("View Retrieved Context"):
                for i, doc in enumerate(docs, start=1):
                    st.markdown(f"### Chunk {i}")
                    st.write(doc.page_content)
                    st.divider()