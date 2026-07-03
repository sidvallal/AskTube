def get_retriever(vector_store, k=5, fetch_k=20):
    """
    Creates an advanced retriever using Max Marginal Relevance (MMR).

    Args:
        vector_store: Chroma vector store instance
        k (int): Final number of chunks returned.
        fetch_k (int): Number of chunks fetched before reranking.

    Returns:
        Retriever object.
    """

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": 0.7
        }
    )

    return retriever


def retrieve_documents(retriever, query: str):
    """
    Retrieves relevant documents for a given query.

    Args:
        retriever: LangChain retriever object
        query (str): User query

    Returns:
        List of relevant documents.
    """

    return retriever.invoke(query)