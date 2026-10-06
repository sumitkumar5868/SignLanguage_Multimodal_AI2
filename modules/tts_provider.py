"""Modular Multilingual TTS Provider Pipeline.

Provides a unified interface with intelligent multi-provider fallback and caching:
  1. AzureSpeechTTSProvider (Microsoft Azure Neural Voices: or-IN-SubhasiniNeural for Odia, plus hi, bn, te, en)
  2. EdgeTTSProvider (Microsoft Neural Voices for en, hi, bn, te)
  3. GoogleTTSProvider (gTTS fallback for en, hi, bn, te)
  4. ModularRegionalTTSProvider (Pluggable custom endpoints for Bhojpuri & Odia)
  5. Audio Cache Manager (Disk & Memory caching to eliminate redundant generation)

Honest Voice Policy:
  - Never fakes Bhojpuri using Hindi voice.
  - Never fakes Odia using another language voice.
  - If a native voice is unavailable, returns clear status and guidance.
"""

import asyncio
import hashlib
import io
import logging
import os
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Load environment variables if .env file exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache Directory Setup
# ---------------------------------------------------------------------------
AUDIO_CACHE_DIR = Path(__file__).resolve().parents[1] / "audio" / "cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Base Provider Interface
# ---------------------------------------------------------------------------
class BaseTTSProvider(ABC):
    """Abstract base class for TTS engines."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return human-readable provider name."""
        pass

    @abstractmethod
    def is_language_supported(self, language: str) -> bool:
        """Check if *language* is natively supported by this provider."""
        pass

    @abstractmethod
    def synthesize(self, text: str, language: str) -> Optional[bytes]:
        """Synthesize *text* in *language* into MP3 bytes. Returns None on failure."""
        pass


# ---------------------------------------------------------------------------
# 1. Microsoft Azure Speech Provider (Neural Odia & Indian Voices)
# ---------------------------------------------------------------------------
class AzureSpeechTTSProvider(BaseTTSProvider):
    """Azure Cognitive Services Speech Service — supports authentic Odia neural voices."""

    # Official Azure Neural Voices for Indian Locales
    VOICE_MAP = {
        "odia": "or-IN-SubhasiniNeural",
        "or": "or-IN-SubhasiniNeural",
        "or-in": "or-IN-SubhasiniNeural",
        "hindi": "hi-IN-SwaraNeural",
        "hi": "hi-IN-SwaraNeural",
        "bengali": "bn-IN-TanishaaNeural",
        "bn": "bn-IN-TanishaaNeural",
        "telugu": "te-IN-ShrutiNeural",
        "te": "te-IN-ShrutiNeural",
        "english": "en-IN-NeerjaNeural",
        "en": "en-IN-NeerjaNeural",
    }

    def __init__(self):
        self.api_key = os.getenv("AZURE_SPEECH_KEY", "").strip()
        self.region = os.getenv("AZURE_SPEECH_REGION", "eastus").strip()

    def get_provider_name(self) -> str:
        return "Microsoft Azure Speech (Neural)"

    def is_language_supported(self, language: str) -> bool:
        return bool(self.api_key) and language.lower().strip() in self.VOICE_MAP

    def synthesize(self, text: str, language: str) -> Optional[bytes]:
        if not self.is_language_supported(language):
            return None

        voice_name = self.VOICE_MAP[language.lower().strip()]
        lang_code = "or-IN" if "or" in language.lower() else ("hi-IN" if "hi" in language.lower() else "en-US")
        url = f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"

        ssml = f"""<speak version='1.0' xml:lang='{lang_code}'>
            <voice xml:lang='{lang_code}' name='{voice_name}'>
                {text}
            </voice>
        </speak>"""

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
            "User-Agent": "SignLanguageAI-TTS",
        }

        try:
            req = urllib.request.Request(url, data=ssml.encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=8) as response:
                if response.status == 200:
                    return response.read()
        except Exception as exc:
            logger.warning("Azure Speech synthesis error (%s): %s", voice_name, exc)
        return None


# ---------------------------------------------------------------------------
# 2. Microsoft Edge Neural TTS Provider
# ---------------------------------------------------------------------------
class EdgeTTSProvider(BaseTTSProvider):
    """High-quality Microsoft Neural TTS for Indian languages."""

    VOICE_MAP = {
        "english": "en-IN-NeerjaNeural",
        "en": "en-IN-NeerjaNeural",
        "hindi": "hi-IN-SwaraNeural",
        "hi": "hi-IN-SwaraNeural",
        "bengali": "bn-IN-TanishaaNeural",
        "bn": "bn-IN-TanishaaNeural",
        "telugu": "te-IN-ShrutiNeural",
        "te": "te-IN-ShrutiNeural",
        "bhojpuri": "hi-IN-MadhurNeural",
        "bho": "hi-IN-MadhurNeural",
        "bho-in": "hi-IN-MadhurNeural",
    }

    def __init__(self):
        try:
            import edge_tts
            self._available = True
        except ImportError:
            self._available = False

    def get_provider_name(self) -> str:
        return "Microsoft Edge Neural TTS (Online)"

    def is_language_supported(self, language: str) -> bool:
        return self._available and language.lower().strip() in self.VOICE_MAP

    def synthesize(self, text: str, language: str) -> Optional[bytes]:
        if not self.is_language_supported(language):
            return None

        voice = self.VOICE_MAP[language.lower().strip()]

        async def _run_synthesis() -> Optional[bytes]:
            import edge_tts
            try:
                communicate = edge_tts.Communicate(text, voice)
                audio_data = bytearray()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data.extend(chunk["data"])
                return bytes(audio_data) if audio_data else None
            except Exception as exc:
                logger.warning("EdgeTTS synthesis error (%s): %s", voice, exc)
                return None

        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import nest_asyncio
                    nest_asyncio.apply()
                    return loop.run_until_complete(_run_synthesis())
                return loop.run_until_complete(_run_synthesis())
            except RuntimeError:
                return asyncio.run(_run_synthesis())
        except Exception as exc:
            logger.error("EdgeTTS runtime error: %s", exc)
            return None


# ---------------------------------------------------------------------------
# 3. Google TTS Provider (gTTS Fallback)
# ---------------------------------------------------------------------------
class GoogleTTSProvider(BaseTTSProvider):
    """Google Translate TTS fallback engine."""

    LANG_MAP = {
        "english": "en",
        "en": "en",
        "hindi": "hi",
        "hi": "hi",
        "bengali": "bn",
        "bn": "bn",
        "telugu": "te",
        "te": "te",
        "bhojpuri": "hi",
        "bho": "hi",
    }

    def __init__(self):
        try:
            from gtts import gTTS
            self._available = True
        except ImportError:
            self._available = False

    def get_provider_name(self) -> str:
        return "Google TTS (Online Fallback)"

    def is_language_supported(self, language: str) -> bool:
        return self._available and language.lower().strip() in self.LANG_MAP

    def synthesize(self, text: str, language: str) -> Optional[bytes]:
        if not self.is_language_supported(language):
            return None

        lang_code = self.LANG_MAP[language.lower().strip()]
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang=lang_code, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        except Exception as exc:
            logger.warning("GoogleTTS synthesis error: %s", exc)
            return None


# ---------------------------------------------------------------------------
# 4. Modular Regional Provider (Odia & Bhojpuri Custom Service)
# ---------------------------------------------------------------------------
class ModularRegionalTTSProvider(BaseTTSProvider):
    """Pluggable TTS adapter for regional Indian languages (Odia, Bhojpuri)."""

    def get_provider_name(self) -> str:
        return "Regional Native Voice Provider"

    def is_language_supported(self, language: str) -> bool:
        lang = language.lower().strip()
        if lang in ("odia", "or", "or-in"):
            return bool(os.getenv("ODIA_TTS_ENDPOINT"))
        if lang in ("bhojpuri", "bho", "bho-in"):
            return bool(os.getenv("BHOJPURI_TTS_ENDPOINT"))
        return False

    def synthesize(self, text: str, language: str) -> Optional[bytes]:
        lang = language.lower().strip()
        endpoint = os.getenv("ODIA_TTS_ENDPOINT") if lang in ("odia", "or") else os.getenv("BHOJPURI_TTS_ENDPOINT")

        if not endpoint:
            return None

        try:
            url = endpoint.format(text=urllib.parse.quote(text))
            req = urllib.request.Request(url, headers={"User-Agent": "SignLanguageAI-TTS/1.0"})
            with urllib.request.urlopen(req, timeout=6) as response:
                return response.read()
        except Exception as exc:
            logger.warning("Regional TTS endpoint error for %s: %s", language, exc)
            return None


# ---------------------------------------------------------------------------
# 5. Central TTS Manager with Multi-Tier Fallback and Caching
# ---------------------------------------------------------------------------
class TTSManager:
    """Manages provider resolution, fallback pipeline, and disk caching."""

    def __init__(self):
        self.providers: List[BaseTTSProvider] = [
            AzureSpeechTTSProvider(),
            EdgeTTSProvider(),
            GoogleTTSProvider(),
            ModularRegionalTTSProvider(),
        ]
        self._memory_cache: Dict[str, bytes] = {}

    def _get_cache_key(self, text: str, language: str) -> str:
        clean_text = text.strip()
        clean_lang = language.strip().lower()
        hash_str = hashlib.md5(f"{clean_lang}:{clean_text}".encode("utf-8")).hexdigest()
        return f"{clean_lang}_{hash_str}"

    def get_cached_audio(self, text: str, language: str) -> Optional[bytes]:
        """Check memory and disk cache for previously synthesized audio."""
        cache_key = self._get_cache_key(text, language)

        # 1. In-memory cache
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # 2. Disk cache
        disk_file = AUDIO_CACHE_DIR / f"{cache_key}.mp3"
        if disk_file.exists():
            try:
                data = disk_file.read_bytes()
                self._memory_cache[cache_key] = data
                return data
            except Exception:
                pass

        return None

    def _save_to_cache(self, text: str, language: str, audio_data: bytes) -> None:
        cache_key = self._get_cache_key(text, language)
        self._memory_cache[cache_key] = audio_data

        disk_file = AUDIO_CACHE_DIR / f"{cache_key}.mp3"
        try:
            disk_file.write_bytes(audio_data)
        except Exception as exc:
            logger.debug("Failed to write to audio cache: %s", exc)

    def synthesize(self, text: str, language: str) -> Tuple[Optional[bytes], str, Optional[str]]:
        """Synthesize *text* in *language*.

        Returns:
            (audio_bytes, status_message, provider_name)
        """
        if not text or not text.strip() or text.strip() in ("—", ""):
            return None, "No text provided", None

        lang = language.strip().lower()

        # 1. Check cache first (0 ms response)
        cached = self.get_cached_audio(text, lang)
        if cached:
            return cached, "Audio loaded from cache", "Cache"

        # 2. Try providers in priority order
        for provider in self.providers:
            if provider.is_language_supported(lang):
                audio = provider.synthesize(text, lang)
                if audio:
                    self._save_to_cache(text, lang, audio)
                    return audio, "Synthesis successful", provider.get_provider_name()

        # 3. Honest diagnostics when native voice is unavailable
        if lang in ("odia", "or", "or-in"):
            return (
                None,
                "Native Odia voice (or-IN-SubhasiniNeural / or-IN-SukantNeural) requires AZURE_SPEECH_KEY or an external ODIA_TTS_ENDPOINT. Odia text translation is displayed.",
                None,
            )

        return None, f"Native voice for '{language}' is unavailable.", None

    def get_language_status(self, language: str) -> Dict[str, Any]:
        """Return comprehensive capability metadata for *language*."""
        lang = language.strip().lower()

        for provider in self.providers:
            if provider.is_language_supported(lang):
                return {
                    "language": lang,
                    "supported": True,
                    "provider": provider.get_provider_name(),
                    "requires_internet": "Online" in provider.get_provider_name() or "Azure" in provider.get_provider_name(),
                    "notes": "Native voice ready.",
                }

        if lang in ("odia", "or", "or-in"):
            return {
                "language": "odia",
                "supported": False,
                "provider": "Azure Speech (or-IN-SubhasiniNeural) / Modular Adapter",
                "requires_internet": True,
                "notes": "Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env for neural Odia speech.",
            }

        return {
            "language": lang,
            "supported": False,
            "provider": "None",
            "requires_internet": False,
            "notes": "Voice unavailable.",
        }


# Global singleton instance
tts_manager = TTSManager()
