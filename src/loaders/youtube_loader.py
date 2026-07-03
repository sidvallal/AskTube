from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
)
from langchain_core.documents import Document


def get_transcript(video_id: str) -> Document:
    """
    Fetches the transcript of a YouTube video and returns it
    as a LangChain Document.

    Tries to fetch transcripts in the following order:
    English -> Hindi -> Any available language
    """

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        try:
            # Prefer English, then Hindi
            transcript = transcript_list.find_transcript(['en', 'hi'])
        except NoTranscriptFound:
            # Fallback to first available transcript
            transcript = next(iter(transcript_list))

        fetched_transcript = transcript.fetch()

        text = " ".join(chunk.text for chunk in fetched_transcript)

        return Document(
            page_content=text,
            metadata={
                "source": f"https://www.youtube.com/watch?v={video_id}",
                "language": transcript.language,
            }
        )

    except TranscriptsDisabled:
        raise Exception("Transcripts are disabled for this video.")

    except NoTranscriptFound:
        raise Exception("No transcript found for this video.")

    except Exception as e:
        raise Exception(f"Failed to fetch transcript: {e}")