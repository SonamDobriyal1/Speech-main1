PHONEME_HINTS = {
    "b": "Press your lips together and use your voice for /b/.",
    "p": "Press your lips together and release a puff of air for /p/.",
    "t": "Tap your tongue just behind your teeth for /t/.",
    "d": "Tap your tongue just behind your teeth and use your voice for /d/.",
    "k": "Lift the back of your tongue and release a quick burst of air for /k/.",
    "g": "Lift the back of your tongue and use your voice for /g/.",
    "r": "Keep your tongue slightly curled for /r/.",
    "l": "Lift the tip of your tongue to the roof just behind your teeth for /l/.",
    "s": "Keep the air moving through a narrow gap for /s/.",
    "z": "Make /z/ by doing /s/ with your voice turned on.",
    "m": "Close your lips and let the sound come through your nose for /m/.",
    "n": "Touch the tongue to the roof of the mouth and let the sound flow through your nose for /n/.",
    "f": "Let air pass through your top teeth and bottom lip for /f/.",
    "v": "Let air pass through your top teeth and bottom lip with voice for /v/.",
    "th": "Place your tongue gently between your teeth and let air out for /th/.",
    "dh": "Place your tongue between your teeth and use your voice for the voiced /th/ sound.",
    "sh": "Round your lips slightly and let the air out softly for /sh/.",
    "ch": "Start with a stop, then release air for /ch/.",
    "j": "Do /ch/ with your voice turned on for /j/.",
    "h": "Let air flow softly from your throat for /h/.",
    "w": "Round your lips and glide into the next sound for /w/.",
    "y": "Start with a small smile and glide into the next sound for /y/.",
    "a": "Open your mouth wide for /a/.",
    "e": "Spread your mouth a little for /e/.",
    "i": "Make a small smile shape for /i/.",
    "o": "Round your lips for /o/.",
    "u": "Round your lips more tightly for /u/.",
    "ai": "Start with /a/ and glide to /i/ for /ai/.",
    "au": "Start with /a/ and glide to /u/ for /au/.",
    "ng": "Let the sound come from the back of your nose for /ng/.",
    "default": "Try shaping your mouth a little differently for that sound.",
}

def _first_mismatch(expected: list[str], spoken: list[str]) -> str | None:
    if not expected:
        return None

    limit = min(len(expected), len(spoken))
    for i in range(limit):
        if expected[i] != spoken[i]:
            return expected[i]

    if len(expected) > len(spoken):
        return expected[limit] if limit < len(expected) else expected[-1]

    if len(expected) < len(spoken):
        return expected[-1]

    return None

def _pair_difference(expected: list[str], contrast: list[str] | None) -> str | None:
    if not contrast:
        return None

    limit = min(len(expected), len(contrast))
    for i in range(limit):
        if expected[i] != contrast[i]:
            return expected[i]

    if len(expected) > len(contrast):
        return expected[limit] if limit < len(expected) else expected[-1]

    return expected[-1] if expected else None

def explain_phoneme(phoneme: str | None, expected_word: str | None = None) -> str:
    if not phoneme:
        return "Good try. Listen carefully and try again."

    hint = PHONEME_HINTS.get(phoneme, PHONEME_HINTS["default"])

    if expected_word:
        return f"{hint} Try it in the word {expected_word}."

    return hint

def generate_feedback(
    expected_word: str,
    spoken_word: str,
    expected_phonemes: list[str],
    spoken_phonemes: list[str],
    contrast_word: str | None = None,
    contrast_phonemes: list[str] | None = None,
) -> dict:
    expected_word = (expected_word or "").strip().lower()
    spoken_word = (spoken_word or "").strip().lower()
    contrast_word = (contrast_word or "").strip().lower()

    if expected_word and spoken_word and spoken_word == expected_word:
        return {
            "correct": True,
            "message": "Great job! That sounded clear and correct.",
            "error_phoneme": None,
            "confused_with_pair": False,
        }

    if contrast_word and spoken_word and spoken_word == contrast_word:
        return {
            "correct": False,
            "message": f"You said '{contrast_word}' instead of '{expected_word}'. Listen for the first sound difference.",
            "error_phoneme": _pair_difference(expected_phonemes, contrast_phonemes),
            "confused_with_pair": True,
        }

    if expected_phonemes and spoken_phonemes and expected_phonemes == spoken_phonemes:
        return {
            "correct": True,
            "message": "Great job! That sounded clear and correct.",
            "error_phoneme": None,
            "confused_with_pair": False,
        }

    mismatch = _first_mismatch(expected_phonemes, spoken_phonemes)
    message = explain_phoneme(mismatch, expected_word)

    return {
        "correct": False,
        "message": message,
        "error_phoneme": mismatch,
        "confused_with_pair": False,
    }