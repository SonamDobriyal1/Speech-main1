import os
import tempfile

from transformers import pipeline

_ASR = None

def _load_model():
    global _ASR
    if _ASR is None:
        print("Loading wav2vec model...")
        _ASR = pipeline(
            "automatic-speech-recognition",
            model="facebook/wav2vec2-base-960h",
        )
    return _ASR

def transcribe_audio(audio_bytes: bytes) -> str:
    if not audio_bytes:
        return ""

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp:
            temp.write(audio_bytes)
            temp_path = temp.name

        asr = _load_model()
        result = asr(temp_path)
        text = result.get("text", "")
        return text.strip().lower()

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass