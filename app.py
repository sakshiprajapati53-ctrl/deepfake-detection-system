import io
import hashlib
import librosa
import numpy as np
import streamlit as st
import torch
import shutil
from pydub import AudioSegment
from PIL import Image
from transformers import (Wav2Vec2FeatureExtractor,AutoModelForAudioClassification)

MODEL_NAME = "mo-thecreator/Deepfake-audio-detection"

ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    AudioSegment.converter = ffmpeg_path
else:
    st.error("FFmpeg not found. Install FFmpeg and restart VS Code.")

st.set_page_config(
    page_title="DeepShield Plus",
    layout="wide"
)

@st.cache_resource
def load_audio_model():
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model

def analyze_audio(audio_bytes):
    # Read MP4/MP3/WAV and convert to mono 16 kHz audio
    audio_segment = AudioSegment.from_file(
        io.BytesIO(audio_bytes)
    )

    audio_segment = audio_segment.set_channels(1)
    audio_segment = audio_segment.set_frame_rate(16000)

    # Convert audio samples to a NumPy array
    audio = np.array(
        audio_segment.get_array_of_samples(),
        dtype=np.float32
    )

    # Normalize samples
    max_value = float(2 ** (8 * audio_segment.sample_width - 1))
    audio = audio / max_value

    sample_rate = 16000

    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    # Analyze at most 30 seconds
    audio = audio[:30 * sample_rate]
    processor, model = load_audio_model()
    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )

    with torch.inference_mode():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=-1)[0]

    results = []
    for index, score in enumerate(scores.tolist()):
        label = model.config.id2label.get(
            index, f"LABEL_{index}"
        )
        results.append({
            "label": str(label),
            "score": float(score),
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )
    return results, len(audio) / sample_rate

st.title("DeepShield Plus")
st.write("AI-Powered Deepfake Detection and Media Forensics")

image_tab, audio_tab = st.tabs(
    ["Image Analysis", "Audio Analysis"]
)

with image_tab:
    st.header("Image Upload")

    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    if image_file is not None:
        image = Image.open(image_file).convert("RGB")
        st.image(image, width=400)

        st.write("Image SHA-256")
        st.code(
            hashlib.sha256(image_file.getvalue()).hexdigest()
        )

        st.info("Image detection model will be connected separately.")

with audio_tab:
    st.header("AI Voice Deepfake Detection")

    audio_file = st.file_uploader(
        "Upload a speech recording",
        type=["wav", "mp3", "flac","mp4","m4a"],
        key="audio_upload",
    )

    if audio_file is not None:
        audio_bytes = audio_file.getvalue()
        st.audio(audio_bytes)

        st.write("Audio SHA-256")
        st.code(hashlib.sha256(audio_bytes).hexdigest())

        if st.button("Analyze Audio", type="primary"):
            try:
                with st.spinner(
                    "Loading AI model and analyzing audio..."
                ):
                    results, duration = analyze_audio(audio_bytes)

                st.metric("Analyzed duration", f"{duration:.2f} seconds")

                st.subheader("Model results")

                for result in results:
                    st.write(
                        f"{result['label']}: "
                        f"{result['score']:.4f}"
                    )
                    st.progress(result["score"])

                top_label = results[0]["label"].lower()

                if "fake" in top_label or "spoof" in top_label:
                    st.warning(
                        "The model leans toward synthetic or spoofed speech."
                    )
                elif "real" in top_label or "bonafide" in top_label:
                    st.success(
                        "The model leans toward genuine speech."
                    )
                else:
                    st.warning(
                        "The model returned an unfamiliar label. "
                        "Check the model's label definitions."
                    )

                st.caption(
                    "Research/demo result only. Scores are model outputs, "
                    "not guaranteed probabilities of authenticity. "
                    "The detector may fail on unfamiliar voices, codecs, "
                    "languages, or synthesis methods."
                )

            except Exception as error:
                st.error(f"Audio analysis failed: {error}")
                # temp line
                st.exception(error)
                st.info(
                    "Check the terminal for details and verify that "
                    "the model downloaded successfully."
                )
