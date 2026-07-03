import streamlit as st

from src.loaders.youtube_loader import get_transcript
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.chunker import split_text

st.set_page_config(page_title="AskTube AI", page_icon="🎥")

st.title("AskTube AI")
st.write("Enter a YouTube Video ID to fetch and process its transcript.")

video_id = st.text_input("Enter YouTube Video ID")

if st.button("Process Video"):
    if not video_id:
        st.warning("Please enter a YouTube Video ID.")
    else:
        try:
            with st.spinner("Fetching transcript..."):
                doc = get_transcript(video_id)

            # Clean transcript
            cleaned_text = clean_text(doc.page_content)
            doc.page_content = cleaned_text

            # Split transcript into chunks
            chunks = split_text(doc)

            st.success("Video processed successfully!")

            # Display metadata
            # st.subheader("📌 Metadata")
            # st.json(doc.metadata)

            # Transcript Preview
            st.subheader("📝 Transcript Preview")
            st.text_area(
                "Transcript",
                doc.page_content[:3000],
                height=300
            )

            # Chunk information
            st.subheader("📦 Chunk Information")
            st.write(f"Total Chunks Created: **{len(chunks)}**")

            # Show first chunk
            st.subheader("🔹 First Chunk")
            st.write(chunks[0].page_content)

            # Optional: Show all chunks
            with st.expander("View All Chunks"):
                for idx, chunk in enumerate(chunks, start=1):
                    st.markdown(f"### Chunk {idx}")
                    st.write(chunk.page_content)
                    st.divider()

        except Exception as e:
            st.error(f"Error: {str(e)}")