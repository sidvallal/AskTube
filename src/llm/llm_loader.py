from langchain_groq import ChatGroq
from src.utils.config import Config

def load_llm():

    if not Config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=Config.GROQ_API_KEY
    )

    return llm