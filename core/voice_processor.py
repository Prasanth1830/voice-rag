"""
core/voice_processor.py — Speech-to-text using OpenAI Whisper API.
Accepts audio file bytes and returns transcribed text.
"""

from __future__ import annotations
import io
import logging
import tempfile
import os

from config import settings

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}


class VoiceProcessor:
    """
    Transcribes audio to text using the OpenAI Whisper API.
    """

    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.OPENAI_WHISPER_MODEL
        logger.info(f"VoiceProcessor initialised (model: {self._model})")

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav", language: str | None = None) -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio file content.
            filename: Original filename (used to infer format).
            language: Optional ISO-639-1 language code (e.g. "en"). Auto-detects if None.

        Returns:
            Transcribed text string.

        Raises:
            ValueError: If the audio format is not supported.
        """
        ext = os.path.splitext(filename)[-1].lower()
        if ext not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported audio format '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}"
            )

        logger.info(f"Transcribing audio: {filename} ({len(audio_bytes)} bytes)")

        # Write to a named temp file so OpenAI SDK can read it
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as audio_file:
                kwargs = {"model": self._model, "file": audio_file}
                if language:
                    kwargs["language"] = language
                transcript = self._client.audio.transcriptions.create(**kwargs)
            text = transcript.text.strip()
            logger.info(f"Transcription: '{text[:100]}...' " if len(text) > 100 else f"Transcription: '{text}'")
            return text
        finally:
            os.unlink(tmp_path)
