from __future__ import annotations

import base64
import logging
from typing import Iterable, List

import numpy as np
from openai import OpenAI

from .detector import DetectedObject

LOGGER = logging.getLogger(__name__)


class CaptioningLLM:
    """Wrapper for OpenAI Vision models that produce concise captions."""

    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI()

    @staticmethod
    def _encode_image(frame: np.ndarray) -> str:
        import cv2

        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("Failed to encode frame as JPEG")
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    @staticmethod
    def _format_detections(objects: Iterable[DetectedObject]) -> str:
        formatted: List[str] = []
        for obj in objects:
            x0, y0, x1, y1 = obj.bbox
            formatted.append(
                f"- {obj.label} ({obj.confidence:.2f}) "
                f"bbox=[{x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f}]"
            )
        return "\n".join(formatted) if formatted else "No detections from Hailo."

    def generate_caption(self, frame: np.ndarray, prompt: str, objects: Iterable[DetectedObject]) -> str:
        """Send the frame and optional detector outputs to the LLM."""
        image_b64 = self._encode_image(frame)
        detection_text = self._format_detections(objects)
        LOGGER.debug("Sending %s to LLM", detection_text.replace("\n", "; "))

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Detection summary:\n{detection_text}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            max_tokens=120,
            temperature=0.3,
        )
        caption = response.choices[0].message.content or ""
        return caption.strip()
