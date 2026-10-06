"""Multilingual Sign Label Mapping & Translation System.

Central translation dictionary mapping sign labels to:
  - English 🇬🇧
  - Hindi 🇮🇳
  - Bhojpuri 🇮🇳
  - Odia 🇮🇳
  - Telugu 🇮🇳
  - Bengali 🇮🇳
"""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Supported Languages Configuration
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES: Dict[str, Dict[str, Any]] = {
    "english": {
        "id": "english",
        "name": "English",
        "native_name": "English",
        "flag": "🇬🇧",
        "code": "en",
        "bcp47": "en-US",
        "tts_supported": True,
    },
    "hindi": {
        "id": "hindi",
        "name": "Hindi",
        "native_name": "हिन्दी",
        "flag": "🇮🇳",
        "code": "hi",
        "bcp47": "hi-IN",
        "tts_supported": True,
    },
    "bhojpuri": {
        "id": "bhojpuri",
        "name": "Bhojpuri",
        "native_name": "भोजपुरी",
        "flag": "🇮🇳",
        "code": "bho",
        "bcp47": "bho-IN",
        "tts_supported": True,
        "tts_note": "Bhojpuri neural speech synthesis ready.",
    },
    "odia": {
        "id": "odia",
        "name": "Odia",
        "native_name": "ଓଡ଼ିଆ",
        "flag": "🇮🇳",
        "code": "or",
        "bcp47": "or-IN",
        "tts_supported": True,
        "tts_note": "Supported via Azure Speech (or-IN-SubhasiniNeural / or-IN-SukantNeural) or custom endpoint.",
    },
    "telugu": {
        "id": "telugu",
        "name": "Telugu",
        "native_name": "తెలుగు",
        "flag": "🇮🇳",
        "code": "te",
        "bcp47": "te-IN",
        "tts_supported": True,
    },
    "bengali": {
        "id": "bengali",
        "name": "Bengali",
        "native_name": "বাংলা",
        "flag": "🇮🇳",
        "code": "bn",
        "bcp47": "bn-IN",
        "tts_supported": True,
    },
}

# Language alias lookup table (maps 'en' -> 'english', 'or' -> 'odia', etc.)
LANGUAGE_ALIASES: Dict[str, str] = {
    "en": "english", "english": "english",
    "hi": "hindi", "hindi": "hindi",
    "bho": "bhojpuri", "bhojpuri": "bhojpuri", "bho-in": "bhojpuri",
    "or": "odia", "odia": "odia", "or-in": "odia", "ory": "odia",
    "te": "telugu", "telugu": "telugu", "te-in": "telugu",
    "bn": "bengali", "bengali": "bengali", "bn-in": "bengali",
}

# ---------------------------------------------------------------------------
# Central Multilingual Translation Dictionary
# ---------------------------------------------------------------------------
SIGN_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # ── Currently Trained Model Signs (5 Signs) ───────────────────────────
    "HELLO": {
        "english": "Hello",
        "hindi": "नमस्ते",
        "bhojpuri": "प्रणाम",
        "odia": "ନମସ୍କାର",
        "telugu": "నమస్కారం",
        "bengali": "নমস্কার",
    },
    "I_LOVE_YOU": {
        "english": "I Love You",
        "hindi": "मैं तुमसे प्यार करता हूँ",
        "bhojpuri": "हम तोहरा से प्यार करीला",
        "odia": "ମୁଁ ତୁମକୁ ଭଲପାଏ",
        "telugu": "నేను నిన్ను ప్రేమిస్తున్నాను",
        "bengali": "আমি তোমাকে ভালোবাসি",
    },
    "I_HATE_YOU": {
        "english": "I Hate You",
        "hindi": "मैं तुमसे नफरत करता हूँ",
        "bhojpuri": "हम तोहरा से नफरत करीला",
        "odia": "ମୁଁ ତୁମକୁ ଘୃଣା କରେ",
        "telugu": "నేను నిన్ను ద్వేషిస్తున్నాను",
        "bengali": "আমি তোমাকে ঘৃণা করি",
    },
    "I_EAT": {
        "english": "I Eat",
        "hindi": "मैं खाता हूँ",
        "bhojpuri": "हम खाईला",
        "odia": "ମୁଁ ଖାଉଛି",
        "telugu": "నేను తింటున్నాను",
        "bengali": "আমি খাচ্ছি",
    },
    "THANK_YOU": {
        "english": "Thank You",
        "hindi": "धन्यवाद",
        "bhojpuri": "धन्यवाद",
        "odia": "ଧନ୍ୟବାଦ",
        "telugu": "ధన్యవాదాలు",
        "bengali": "ধন্যবাদ",
    },

    # ── Planned Vocabulary (For Future Dataset Expansion) ─────────────────
    "HELP": {
        "english": "Help",
        "hindi": "मदद",
        "bhojpuri": "मदद",
        "odia": "ସାହାଯ୍ୟ",
        "telugu": "సహాయం",
        "bengali": "সাহায্য",
    },
    "WATER": {
        "english": "Water",
        "hindi": "पानी",
        "bhojpuri": "पानी",
        "odia": "ପାଣି",
        "telugu": "నీరు",
        "bengali": "জল",
    },
    "FOOD": {
        "english": "Food",
        "hindi": "खाना",
        "bhojpuri": "खाना",
        "odia": "ଖାଦ୍ୟ",
        "telugu": "ఆహారం",
        "bengali": "খাবার",
    },
    "BATHROOM": {
        "english": "Bathroom",
        "hindi": "बाथरूम",
        "bhojpuri": "शौचालय",
        "odia": "ଶୌଚାଳୟ",
        "telugu": "స్నାନాల గది",
        "bengali": "বাথরুম",
    },
    "PAIN": {
        "english": "Pain",
        "hindi": "दर्द",
        "bhojpuri": "दरद",
        "odia": "ଯନ୍ତ୍ରଣା",
        "telugu": "నొప్పి",
        "bengali": "ব্যথা",
    },
    "MEDICINE": {
        "english": "Medicine",
        "hindi": "दवाई",
        "bhojpuri": "दवाई",
        "odia": "ଔଷଧ",
        "telugu": "మందు",
        "bengali": "ওষুধ",
    },
    "YES": {
        "english": "Yes",
        "hindi": "हाँ",
        "bhojpuri": "हाँ",
        "odia": "ହଁ",
        "telugu": "అవును",
        "bengali": "হ্যাঁ",
    },
    "NO": {
        "english": "No",
        "hindi": "नहीं",
        "bhojpuri": "ना",
        "odia": "ନାହିଁ",
        "telugu": "కాదు",
        "bengali": "না",
    },
    "EMERGENCY": {
        "english": "Emergency",
        "hindi": "आपातकाल",
        "bhojpuri": "आपातकाल",
        "odia": "ଜରୁରୀକାଳୀନ",
        "telugu": "అత్యవసర పరిస్థితి",
        "bengali": "জরুরী অবস্থা",
    },
    "CALL_DOCTOR": {
        "english": "Call Doctor",
        "hindi": "डॉक्टर को बुलाओ",
        "bhojpuri": "डॉक्टर के बोलावल जाव",
        "odia": "ଡାକ୍ତରଙ୍କୁ ଡାକନ୍ତୁ",
        "telugu": "డాక్టర్‌ని పిలవండి",
        "bengali": "ডাক্তার ডাকুন",
    },
    "SORRY": {
        "english": "Sorry",
        "hindi": "माफ़ कीजिये",
        "bhojpuri": "माफ करीं",
        "odia": "କ୍ଷମା କରନ୍ତୁ",
        "telugu": "క్షమించండి",
        "bengali": "দুঃখিত",
    },
    "PLEASE": {
        "english": "Please",
        "hindi": "कृपया",
        "bhojpuri": "कृपया",
        "odia": "ଦୟାକରି",
        "telugu": "దయచేసి",
        "bengali": "অনুগ্রহ করে",
    },
    "GOODBYE": {
        "english": "Goodbye",
        "hindi": "अलविदा",
        "bhojpuri": "अलविदा",
        "odia": "ବିଦାୟ",
        "telugu": "వీడ్కోలు",
        "bengali": "বিদায়",
    },
    "REST": {
        "english": "Rest",
        "hindi": "आराम",
        "bhojpuri": "आराम",
        "odia": "ବିଶ୍ରାମ",
        "telugu": "విశ్రాంతి",
        "bengali": "বিশ্রাম",
    },
    "SICK": {
        "english": "Sick",
        "hindi": "बीमार",
        "bhojpuri": "बीमार",
        "odia": "ଅସୁସ୍ଥ",
        "telugu": "అనారోగ్యం",
        "bengali": "অসুস্থ",
    },
    "SLEEP": {
        "english": "Sleep",
        "hindi": "सोना",
        "bhojpuri": "सुतल",
        "odia": "ଶୋଇବା",
        "telugu": "నిద్ర",
        "bengali": "ঘুমানো",
    },
    "STOP": {
        "english": "Stop",
        "hindi": "रुकिए",
        "bhojpuri": "रुकीं",
        "odia": "ଅଟକନ୍ତୁ",
        "telugu": "ఆగండి",
        "bengali": "থামুন",
    },
}

SUPPORTED_LABELS = frozenset(SIGN_TRANSLATIONS.keys())

_FALLBACK_TRANSLATIONS: Dict[str, str] = {
    "english": "Translation unavailable",
    "hindi": "अनुवाद अनुपलब्ध",
    "bhojpuri": "अनुवाद उपलब्ध नइखे",
    "odia": "ଅନୁବାଦ ଉପଲବ୍ଧ ନାହିଁ",
    "telugu": "అనువాదం అందుబాటులో లేదు",
    "bengali": "অনুবাদ উপলব্ধ নেই",
}


def normalize_language_key(language: Optional[str]) -> str:
    """Resolve language alias (e.g. 'or' -> 'odia', 'hi' -> 'hindi')."""
    if not language:
        return "english"
    clean = language.strip().lower()
    return LANGUAGE_ALIASES.get(clean, "english")


def get_all_translations(label: Optional[str]) -> Dict[str, str]:
    """Return dictionary of translations for all supported languages."""
    if not label:
        return {lang: "—" for lang in SUPPORTED_LANGUAGES}

    key = label.strip().upper()
    if key in SIGN_TRANSLATIONS:
        result = dict(SIGN_TRANSLATIONS[key])
        # Also populate short codes for maximum client compatibility
        for short_code, standard_name in [("en", "english"), ("hi", "hindi"), ("bho", "bhojpuri"), ("or", "odia"), ("te", "telugu"), ("bn", "bengali")]:
            result[short_code] = result.get(standard_name, "—")
        return result

    return dict(_FALLBACK_TRANSLATIONS)


def get_translation(label: Optional[str], language: str = "english") -> str:
    """Return translation for a specific sign label in the given language."""
    lang_key = normalize_language_key(language)
    translations = get_all_translations(label)
    return translations.get(lang_key, translations.get("english", "Translation unavailable"))


def get_sign_info(label: Optional[str]) -> tuple:
    """Backwards-compatible helper returning (English text, Hindi text)."""
    trans = get_all_translations(label)
    return (trans.get("english", "—"), trans.get("hindi", "—"))


def get_supported_languages_list() -> List[Dict[str, Any]]:
    """Return metadata list of all supported languages for the frontend."""
    return list(SUPPORTED_LANGUAGES.values())
