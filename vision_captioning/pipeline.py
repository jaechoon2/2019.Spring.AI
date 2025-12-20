from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from .config import CaptioningConfig
from .detector import BaseDetector, DetectionResult
from .llm_client import CaptioningLLM
from .tts_client import TTSClient

LOGGER = logging.getLogger(__name__)


class VisionCaptioningPipeline:
    """Capture → Detect → Caption → TTS loop."""

    def __init__(
        self,
        config: CaptioningConfig,
        detector: BaseDetector,
        llm_client: Optional[CaptioningLLM],
        tts_client: Optional[TTSClient],
    ):
        self.config = config
        self.detector = detector
        self.llm_client = llm_client
        self.tts_client = tts_client

    def _open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open camera index {self.config.camera_index}")
        if self.config.frame_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        if self.config.frame_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        return cap

    def _capture_frame(self, cap: cv2.VideoCapture) -> np.ndarray:
        success, frame = cap.read()
        if not success or frame is None:
            raise RuntimeError("Failed to capture frame from camera")
        return frame

    def _caption_frame(self, frame: np.ndarray, detection: DetectionResult) -> str:
        if self.llm_client and self.config.llm_enabled:
            return self.llm_client.generate_caption(
                frame=frame,
                prompt=self.config.prompt,
                objects=detection.objects,
            )

        if detection.objects:
            labels = ", ".join(
                f"{obj.label} ({obj.confidence:.2f})" for obj in detection.objects
            )
            return f"Detected objects: {labels}"

        return "No detections available."

    def _speak_if_needed(self, caption: str) -> None:
        if self.config.speak and self.tts_client:
            self.tts_client.speak(caption)

    def run(self) -> None:
        LOGGER.info("Starting vision captioning pipeline. Press Ctrl+C to exit.")
        cap = self._open_camera()
        try:
            while True:
                try:
                    frame = self._capture_frame(cap)
                    detection = self.detector.detect(frame)
                    caption = self._caption_frame(frame, detection)
                    LOGGER.info("Caption: %s", caption)
                    self._speak_if_needed(caption)
                except KeyboardInterrupt:
                    LOGGER.info("Stopping pipeline on user interrupt.")
                    break
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Pipeline error: %s", exc)
                time.sleep(self.config.interval_seconds)
        finally:
            cap.release()
