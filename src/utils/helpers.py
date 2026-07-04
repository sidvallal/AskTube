import re


def extract_video_id(url: str) -> str:
    """
    Extracts YouTube video ID from a URL.
    """

    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return url.strip()


def truncate_text(text: str, max_length: int = 300):
    """
    Truncates long text for preview purposes.
    """

    if len(text) <= max_length:
        return text

    return text[:max_length] + "..."