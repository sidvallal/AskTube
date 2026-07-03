from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from src.prompts.qa_prompt import QA_PROMPT


def build_rag_chain(llm, retriever):
    """
    Builds and returns the RAG chain.
    """

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=QA_PROMPT
    )

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

    return rag_chain