from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)
from langchain_classic.chains import create_retrieval_chain
from langchain_core.runnables import RunnableLambda

from src.prompts.qa_prompt import QA_PROMPT


def build_rag_chain(llm, retriever):
    """
    Builds the RAG chain with explicit question extraction.
    """

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=QA_PROMPT
    )

    # Explicitly pass the question string to the retriever
    question_retriever = RunnableLambda(
        lambda x: retriever.invoke(x["input"])
    )

    rag_chain = create_retrieval_chain(
        retriever=question_retriever,
        combine_docs_chain=document_chain
    )

    rag_chain = rag_chain.with_config(
        run_name="AskTube RAG Chain",
        tags=["AskTube", "RAG"],
        metadata={"project": "AskTube-AI"}
    )

    return rag_chain