# Program title: Image to Audio Storytelling App

import streamlit as st
from transformers import pipeline

# Load models once
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


# Image → Caption
def img2text(image_path):
    caption = img_model(image_path)[0]["generated_text"]
    return caption


# Caption → Story
def text2story(text):

    prompt = f"""
    Write a simple and friendly story for children aged 3 to 10.
    Use easy words and a happy tone.
    The story should be between 50 and 100 words.

    Scenario: {text}
    """

    story = story_model(
        prompt,
        max_new_tokens=120,
        do_sample=True
    )[0]["generated_text"]

    words = story.split()

    if len(words) > 100:
        story = " ".join(words[:100])

    if len(words) < 50:
        story += " The adventure continued happily."

    return story


# Story → Audio
def text2audio(story):
    audio_data = audio_model(story)
    return audio_data


# Main Streamlit App
def main():

    st.set_page_config(page_title="Image to Audio Story", page_icon="🦜")

    st.header("Turn Your Image into an Audio Story")

    uploaded_file = st.file_uploader(
        "Select an Image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file,
                 caption="Uploaded Image",
                 use_column_width=True)

        # Stage 1
        st.text("Processing image...")
        scenario = img2text("temp_image.png")
        st.write("Caption:", scenario)

        # Stage 2
        st.text("Generating story...")
        story = text2story(scenario)
        st.write(story)

        # Stage 3
        if st.button("Play Audio"):

            st.text("Generating audio...")
            audio_data = text2audio(story)

            audio_array = audio_data["audio"]
            sample_rate = audio_data["sampling_rate"]

            st.audio(audio_array,
                     sample_rate=sample_rate)


if __name__ == "__main__":
    main()
story = story_generator(prompt, return_full_text=False)[0]["generated_text"]
