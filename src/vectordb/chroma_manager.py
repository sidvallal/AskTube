from langchain_chroma import Chroma


def create_vector_store(chunks, embedding_model):
    """
    Creates an in-memory Chroma vector store for the current video.

    Args:
        chunks: List of LangChain Document chunks.
        embedding_model: HuggingFace embedding model.

    Returns:
        Chroma vector store.
    """

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vector_store