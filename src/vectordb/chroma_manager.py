from langchain_chroma import Chroma

CHROMA_DB_PATH = "chroma_db"


def create_vector_store(chunks, embedding_model):
    """
    Creates a Chroma vector store from document chunks
    and persists it locally.
    """

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH
    )

    return vector_store


def load_vector_store(embedding_model):
    """
    Loads an existing Chroma vector store.
    """

    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_model
    )

    return vector_store