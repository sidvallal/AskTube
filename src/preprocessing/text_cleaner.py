import re


def clean_text(text: str) -> str:
    """
    Cleans transcript text by removing extra spaces,
    newlines, and unwanted characters.
    """

    # Remove newline characters
    text = text.replace("\n", " ")

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove timestamps like [Music], [Applause]
    text = re.sub(r"\[.*?\]", "", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text