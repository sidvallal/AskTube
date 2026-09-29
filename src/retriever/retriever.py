def get_retriever(vector_store, k=6, fetch_k=30):
    """
    Creates an MMR retriever with LangSmith tracing configuration.
    """

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": 0.5
        }
    )

    # Add LangSmith tracing configuration
    retriever = retriever.with_config(
        run_name="AskTube MMR Retriever",
        tags=["AskTube", "MMR", "Retrieval"],
        metadata={
            "k": k,
            "fetch_k": fetch_k,
            "search_type": "mmr"
        }
    )

    return retriever


def retrieve_documents(retriever, query: str):
    """
    Retrieves relevant documents for a given query.
    """

    return retriever.invoke(
        query,
        config={
            "run_name": "Retrieve YouTube Chunks"
        }
    )