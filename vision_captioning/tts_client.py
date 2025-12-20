from __future__ import annotations

import logging

import pyttsx3

LOGGER = logging.getLogger(__name__)


class TTSClient:
    """Lightweight wrapper around pyttsx3 for offline speech synthesis."""

    def __init__(self, rate: int = 170):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)

    def speak(self, text: str) -> None:
        LOGGER.info("Speaking caption: %s", text)
        self.engine.say(text)
        self.engine.runAndWait()

    def stop(self) -> None:
        self.engine.stop()
