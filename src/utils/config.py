import os
from dotenv import load_dotenv

load_dotenv()


class Config:

    # Groq Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # LangSmith Configuration
    LANGSMITH_TRACING = os.getenv(
        "LANGSMITH_TRACING", "true"
    )

    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

    LANGSMITH_PROJECT = os.getenv(
        "LANGSMITH_PROJECT", "AskTube-AI"
    )