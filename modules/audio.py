"""Multilingual Audio Engine.

Coordinates speech synthesis through the modular TTS Provider Pipeline
and plays audio asynchronously.

Public API:
    speak_text(text: str, language: str = "hindi") -> Tuple[bool, str]
    speak_hindi(text: str) -> None  # Backwards-compatible alias
"""

import io
import logging
import os
import tempfile
import threading
import time
from typing import Optional, Tuple

from modules.tts_provider import tts_manager

logger = logging.getLogger(__name__)

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False
    logger.warning("pygame not installed. Local audio playback disabled.")



_mixer_initialised = False
_last_spoken_key: str = ""
_last_spoken_time: float = 0.0
_DEBOUNCE_SECONDS: float = 2.0
_audio_lock = threading.Lock()


def _ensure_mixer() -> bool:
    """Initialise pygame.mixer once. Returns True if ready."""
    global _mixer_initialised
    if _mixer_initialised:
        return True
    if not _PYGAME_AVAILABLE:
        return False
    try:
        pygame.mixer.init()
        _mixer_initialised = True
        return True
    except Exception as exc:
        logger.error("Failed to initialise pygame.mixer: %s", exc)
        return False


def _play_audio_bytes(audio_data: bytes) -> None:
    """Play audio bytes in a background thread."""
    if not _ensure_mixer() or not audio_data:
        return

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
    except Exception as exc:
        logger.error("Audio playback error: %s", exc)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def speak_text(text: str, language: str = "hindi") -> Tuple[bool, str]:
    """Synthesize and speak *text* in *language* asynchronously.

    Returns:
        (success: bool, message: str)
    """
    global _last_spoken_key, _last_spoken_time

    if not text or not text.strip() or text.strip() in ("—", ""):
        return False, "No text provided"

    clean_text = text.strip()
    clean_lang = language.strip().lower()

    # Synthesize audio via TTS Provider Pipeline (with caching)
    audio_bytes, message, provider = tts_manager.synthesize(clean_text, clean_lang)

    if not audio_bytes:
        return False, message

    # Debounce checks
    now = time.time()
    call_key = f"{clean_lang}:{clean_text}"

    with _audio_lock:
        if call_key == _last_spoken_key and (now - _last_spoken_time) < _DEBOUNCE_SECONDS:
            return True, "Debounced"
        _last_spoken_key = call_key
        _last_spoken_time = now

    thread = threading.Thread(target=_play_audio_bytes, args=(audio_bytes,), daemon=True)
    thread.start()
    return True, f"Speaking {language} ({provider})"


def speak_hindi(text: str) -> None:
    """Backwards-compatible convenience helper."""
    speak_text(text, language="hindi")


def is_audio_available() -> bool:
    """Return True if local playback is supported."""
    return _PYGAME_AVAILABLE and _ensure_mixer()
