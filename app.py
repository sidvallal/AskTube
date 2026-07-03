import streamlit as st

from src.loaders.youtube_loader import get_transcript
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.chunker import split_text
from src.embeddings.embedding_model import load_embedding_model
from src.vectordb.chroma_manager import create_vector_store
from src.retriever.retriever import get_retriever
from src.llm.llm_loader import load_llm
from src.llm.qa_chain import build_rag_chain

st.set_page_config(
    page_title="AskTube AI",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 AskTube AI")
st.markdown(
    "Chat with any YouTube video by simply providing its Video ID."
)

# ---------------------------
# Video Processing Section
# ---------------------------

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
            document.page_content = clean_text(
                document.page_content
            )

            # Split transcript
            chunks = split_text(document)

            # Load embeddings
            embedding_model = load_embedding_model()

            # Create vector store
            vector_store = create_vector_store(
                chunks=chunks,
                embedding_model=embedding_model
            )

            # Create retriever
            retriever = get_retriever(vector_store)

            # Load LLM
            llm = load_llm()

            # Build RAG Chain
            rag_chain = build_rag_chain(
                llm=llm,
                retriever=retriever
            )

            # Save chain in session state
            st.session_state.rag_chain = rag_chain

        st.success("Video processed successfully!")

        # Metrics
        col1, col2 = st.columns(2)

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

        st.info(
            "The video is now ready for question answering."
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")


# ---------------------------
# Chat Section
# ---------------------------

if "rag_chain" in st.session_state:

    st.divider()
    st.subheader("💬 Ask Questions")

    query = st.text_input(
        "Enter your question",
        placeholder="What is this video about?"
    )

    if st.button("Get Answer", use_container_width=True):

        if not query:
            st.warning("Please enter a question.")
            st.stop()

        try:
            with st.spinner("Generating answer..."):

                response = st.session_state.rag_chain.invoke(
                    {"input": query}
                )

            st.subheader("Answer")
            st.write(response["answer"])

            # with st.expander("Retrieved Context"):
            #     for idx, doc in enumerate(response["context"], start=1):
            #         st.markdown(f"### Chunk {idx}")
            #         st.write(doc.page_content)
            #         st.divider()

        except Exception as e:
            st.error(f"Error: {str(e)}")