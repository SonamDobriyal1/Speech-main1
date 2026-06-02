from fastapi import APIRouter, UploadFile, File, Form

from services.wav2vec_service import transcribe_audio
from services.phoneme_service import get_phonemes, extract_first_word
from services.feedback_service import generate_feedback

router = APIRouter()

@router.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    expected_word: str = Form(...),
    contrast_word: str | None = Form(None),
):
    try:
        audio_bytes = await file.read()

        transcription = transcribe_audio(audio_bytes)
        expected_word = (expected_word or "").strip().lower()
        contrast_word = (contrast_word or "").strip().lower() if contrast_word else None

        spoken_word = extract_first_word(transcription, expected_word)

        expected_phonemes = get_phonemes(expected_word)
        spoken_phonemes = get_phonemes(spoken_word)
        contrast_phonemes = get_phonemes(contrast_word) if contrast_word else None

        feedback_data = generate_feedback(
            expected_word=expected_word,
            spoken_word=spoken_word,
            expected_phonemes=expected_phonemes,
            spoken_phonemes=spoken_phonemes,
            contrast_word=contrast_word,
            contrast_phonemes=contrast_phonemes,
        )

        return {
            "transcription": transcription,
            "spoken_word": spoken_word,
            "feedback": feedback_data["message"],
            "correct": feedback_data["correct"],
            "error_phoneme": feedback_data["error_phoneme"],
            "confused_with_pair": feedback_data["confused_with_pair"],
            "expected_word": expected_word,
        }

    except Exception as e:
        print("ERROR:", e)
        return {
            "transcription": "",
            "spoken_word": "",
            "feedback": "Error processing audio",
            "correct": False,
            "error_phoneme": None,
            "confused_with_pair": False,
            "expected_word": expected_word if "expected_word" in locals() else "",
        }