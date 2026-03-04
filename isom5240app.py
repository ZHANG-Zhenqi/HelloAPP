# app.py
# Program title: Image to Audio Storytelling App (50–100 words, kid-friendly)

import re
import uuid
from pathlib import Path

import streamlit as st
from transformers import pipeline


# -----------------------------
# Text helpers
# -----------------------------
def clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def sanitize_story(s: str) -> str:
    """
    Remove common junk the model may output:
    - prompt-like instructions
    - parentheses content (e.g., actor names)
    - movie-summary tone openers
    """
    s = clean_whitespace(s)

    # remove parentheses content like (Anjali Devi)
    s = re.sub(r"\([^)]*\)", "", s)
    s = clean_whitespace(s)

    # remove some common instruction-ish beginnings
    s = re.sub(
        r"^(write a|use easy|the story should|scenario:|scene:|story:|instructions:).*?$",
        "",
        s,
        flags=re.I,
    )
    s = clean_whitespace(s)

    # remove "The story begins..." style movie-summary openers
    s = re.sub(
        r"^(the story (begins|starts)|the story takes).*?\.\s*",
        "",
        s,
        flags=re.I,
    )
    s = clean_whitespace(s)

    return s


def enforce_word_range(story: str, min_words: int = 50, max_words: int = 100) -> str:
    story = sanitize_story(story)
    words = story.split()

    # truncate
    if len(words) > max_words:
        story = " ".join(words[:max_words]).rstrip(" ,;:") + "."
        words = story.split()

    # pad (simple safe ending)
    if len(words) < min_words:
        story = (story + " Everyone smiled, said thank you, and went home happily.").strip()
        words = story.split()
        if len(words) > max_words:
            story = " ".join(words[:max_words]).rstrip(" ,;:") + "."

    return story


# -----------------------------
# Load models once
# -----------------------------
@st.cache_resource
def load_models():
    img_model = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )

    story_model = pipeline(
        "text-generation",
        model="pranavpsv/genre-story-generator-v2"
    )

    audio_model = pipeline(
        "text-to-audio",
        model="Matthijs/mms-tts-eng"
    )

    return img_model, story_model, audio_model


img_model, story_model, audio_model = load_models()


# -----------------------------
# Pipeline stages
# -----------------------------
def img2caption(image_path: str) -> str:
    caption = img_model(image_path)[0]["generated_text"]
    return clean_whitespace(caption)


def caption2story(caption: str) -> str:
    # Short, strict prompt reduces "requirements printed as story"
    prompt = (
        "Write a kids story for ages 3 to 10. "
        "Use simple words and short sentences. "
        "50 to 100 words. "
        "No character lists. No actor names. No parentheses. "
        "Do not repeat sentences. "
        f"Scene: {caption} "
        "Story:"
    )

    gen = story_model(
        prompt,
        max_new_tokens=160,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        return_full_text=False,   # IMPORTANT: do not include prompt in output
    )[0]["generated_text"]

    story = enforce_word_range(gen, 50, 100)
    return story


def story2audio(story: str):
    return audio_model(story)


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Image to Audio Story", page_icon="🦜")
    st.header("Turn Your Image into an Audio Story")

    uploaded = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload a JPG or PNG image to start.")
        return

    # Save to safe temp filename
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}.png"
    tmp_path.write_bytes(uploaded.getbuffer())

    st.image(uploaded, caption="Uploaded image", use_column_width=True)

    # Stage 1: Caption
    with st.spinner("Processing image..."):
        caption = img2caption(str(tmp_path))
    st.write("Caption:", caption)

    # Stage 2: Story
    with st.spinner("Generating story..."):
        story = caption2story(caption)

    st.subheader("Story (50–100 words)")
    st.write(story)

    # Stage 3: Audio
    if st.button("Play Audio"):
        with st.spinner("Generating audio..."):
            audio = story2audio(story)
        st.audio(audio["audio"], sample_rate=audio["sampling_rate"])


if __name__ == "__main__":
    main()
