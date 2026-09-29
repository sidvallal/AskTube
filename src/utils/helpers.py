import re

import requests


def extract_video_id(url: str) -> str:
    """
    Extracts YouTube video ID from a URL.
    """

    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return url.strip()


def truncate_text(text, max_length: int = 300) -> str:
    """
    Truncates text for preview purposes.
    Handles dictionaries and other non-string values safely.
    """

    if not isinstance(text, str):
        text = str(text)

    if len(text) <= max_length:
        return text

    return text[:max_length] + "..."


def get_video_title(video_id: str) -> str:
    """
    Fetches the video title using YouTube's public oEmbed endpoint.
    No API key required. Falls back to the video ID if the request fails.
    """

    try:
        oembed_url = (
            "https://www.youtube.com/oembed"
            f"?url=https://www.youtube.com/watch?v={video_id}&format=json"
        )
        response = requests.get(oembed_url, timeout=5)
        response.raise_for_status()
        return response.json().get("title", video_id)

    except Exception:
        return video_id