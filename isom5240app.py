import re
import uuid
from pathlib import Path

import streamlit as st
from transformers import pipeline


# -------------------------
# Utils
# -------------------------
def clean_text(s: str) -> str:
    # 清理多余空白
    s = re.sub(r"\s+", " ", s).strip()

    # 去掉常见“指令残留”前缀
    # 这些不是故事正文
    bad_prefixes = [
        "write a simple",
        "write a short",
        "use easy words",
        "the story should",
        "scenario:",
        "scene:",
        "story:",
        "instructions:",
        "generating story",
    ]
    lowered = s.lower()
    for bp in bad_prefixes:
        if lowered.startswith(bp):
            # 删除第一句到第一个句号为止
            if "." in s:
                s = s.split(".", 1)[1].strip()
            break

    # 再次压缩空白
    s = re.sub(r"\s+", " ", s).strip()
    return s


def enforce_word_range(story: str, min_words: int = 50, max_words: int = 100) -> str:
    story = clean_text(story)
    words = story.split()

    # 太长直接截断到 max_words
    if len(words) > max_words:
        story = " ".join(words[:max_words]).rstrip(" ,;:") + "."
        words = story.split()

    # 太短就补写，直到达到 min_words
    # 用一次续写补到位，避免一直循环
    if len(words) < min_words:
        need = min_words - len(words)
        return story, need
    return story, 0


# -------------------------
# Load models once
# -------------------------
@st.cache_resource
def load_models():
    img_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    story_model = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    audio_model = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    return img_model, story_model, audio_model


img_model, story_model, audio_model = load_models()


# -------------------------
# Pipeline stages
# -------------------------
def img2caption(image_path: str) -> str:
    out = img_model(image_path)[0]["generated_text"]
    return clean_text(out)


def caption2story(caption: str) -> str:
    # prompt 写短一点，更稳
    prompt = (
        "Write a kid-friendly story for ages 3 to 10. "
        "Use simple words. Keep it safe and happy. "
        "Write 50 to 100 words. "
        f"Scene: {caption}\nStory:"
    )

    gen = story_model(
        prompt,
        max_new_tokens=160,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        return_full_text=False,  # 重点：只返回生成部分，不带 prompt
    )[0]["generated_text"]

    story = clean_text(gen)
    story, need = enforce_word_range(story, 50, 100)

    # 不足 50 words：续写补齐
    if need > 0:
        continuation_prompt = (
            f"{story}\nContinue the same story in simple words. "
            f"Add about {need + 10} words. "
            "Do not repeat any instructions."
        )
        more = story_model(
            continuation_prompt,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            return_full_text=False,
        )[0]["generated_text"]

        story = clean_text(story + " " + more)
        # 再次强行卡在 50–100
        words = story.split()
        if len(words) > 100:
            story = " ".join(words[:100]).rstrip(" ,;:") + "."
        elif len(words) < 50:
            # 如果还是不够，补一句固定短句兜底
            story = (story + " They smiled, held hands, and went home together.").strip()
            words = story.split()
            if len(words) > 100:
                story = " ".join(words[:100]).rstrip(" ,;:") + "."

    return story


def story2audio(story: str):
    return audio_model(story)


# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(page_title="Image to Audio Story", page_icon="🦜")
    st.header("Turn Your Image into an Audio Story")

    uploaded = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload a JPG or PNG image to start.")
        return

    # 保存成安全文件名，避免覆盖和奇怪字符
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}.png"
    tmp_path.write_bytes(uploaded.getbuffer())

    st.image(uploaded, caption="Uploaded image", use_column_width=True)

    with st.spinner("Processing image..."):
        caption = img2caption(str(tmp_path))
    st.write("Caption:", caption)

    with st.spinner("Generating story..."):
        story = caption2story(caption)

    st.subheader("Story (50–100 words)")
    st.write(story)

    if st.button("Play Audio"):
        with st.spinner("Generating audio..."):
            audio = story2audio(story)
        st.audio(audio["audio"], sample_rate=audio["sampling_rate"])


if __name__ == "__main__":
    main()
