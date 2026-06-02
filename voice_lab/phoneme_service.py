import re

try:
    import eng_to_ipa as ipa
except Exception:
    ipa = None

WORD_PHONEMES = {
    "cat": ["k", "a", "t"],
    "bat": ["b", "a", "t"],
    "pat": ["p", "a", "t"],
    "big": ["b", "i", "g"],
    "pig": ["p", "i", "g"],
    "bin": ["b", "i", "n"],
    "pin": ["p", "i", "n"],
    "bit": ["b", "i", "t"],
    "pit": ["p", "i", "t"],
    "back": ["b", "a", "k"],
    "pack": ["p", "a", "k"],
    "ban": ["b", "a", "n"],
    "pan": ["p", "a", "n"],
    "tab": ["t", "a", "b"],
    "tap": ["t", "a", "p"],
    "slab": ["s", "l", "a", "b"],
    "slap": ["s", "l", "a", "p"],
    "cub": ["k", "u", "b"],
    "cup": ["k", "u", "p"],
    "mob": ["m", "o", "b"],
    "mop": ["m", "o", "p"],
    "rib": ["r", "i", "b"],
    "rip": ["r", "i", "p"],

    "right": ["r", "i", "t"],
    "light": ["l", "i", "t"],
    "road": ["r", "o", "d"],
    "load": ["l", "o", "d"],
    "rock": ["r", "o", "k"],
    "lock": ["l", "o", "k"],
    "red": ["r", "e", "d"],
    "led": ["l", "e", "d"],
    "rice": ["r", "i", "s"],
    "lice": ["l", "i", "s"],
    "car": ["k", "a", "r"],
    "cal": ["k", "a", "l"],
    "star": ["s", "t", "a", "r"],
    "stall": ["s", "t", "a", "l"],
    "far": ["f", "a", "r"],
    "fall": ["f", "o", "l"],

    "think": ["th", "i", "n", "k"],
    "thin": ["th", "i", "n"],
    "thumb": ["th", "u", "m", "b"],
    "three": ["th", "r", "e"],
    "bath": ["b", "a", "th"],
    "bass": ["b", "a", "s"],
    "sink": ["s", "i", "n", "k"],
    "sin": ["s", "i", "n"],
    "some": ["s", "u", "m"],
    "tree": ["t", "r", "e"],
    "this": ["dh", "i", "s"],
    "dis": ["d", "i", "s"],
    "that": ["dh", "a", "t"],
    "dat": ["d", "a", "t"],
    "then": ["dh", "e", "n"],
    "den": ["d", "e", "n"],
    "those": ["dh", "o", "z"],
    "doze": ["d", "o", "z"],
    "the": ["dh", "e"],
    "duh": ["d", "u"],

    "coat": ["k", "o", "t"],
    "goat": ["g", "o", "t"],
    "cap": ["k", "a", "p"],
    "gap": ["g", "a", "p"],
    "came": ["k", "e", "m"],
    "game": ["g", "e", "m"],
    "cold": ["k", "o", "l", "d"],
    "gold": ["g", "o", "l", "d"],
    "cane": ["k", "a", "n"],
    "gain": ["g", "a", "n"],

    "dog": ["d", "o", "g"],
    "sun": ["s", "u", "n"],
    "bag": ["b", "a", "g"],
    "book": ["b", "u", "k"],
    "go": ["g", "o"],
    "gum": ["g", "u", "m"],
}

def extract_first_word(text: str, expected_word: str | None = None) -> str:
    if not text:
        return ""

    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return ""

    if expected_word:
        expected = expected_word.lower().strip()
        if expected in words:
            return expected

    return words[0]

def _fallback_phonemes(word: str) -> list[str]:
    if ipa is None:
        return list(word)

    try:
        ipa_text = ipa.convert(word)
        ipa_text = re.sub(r"\s+", "", ipa_text)
        if not ipa_text:
            return list(word)
        return list(ipa_text)
    except Exception:
        return list(word)

def get_phonemes(text: str) -> list[str]:
    word = extract_first_word(text)
    if not word:
        return []

    if word in WORD_PHONEMES:
        return WORD_PHONEMES[word]

    return _fallback_phonemes(word)