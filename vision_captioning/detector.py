from __future__ import annotations

import importlib.util
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]


@dataclass
class DetectionResult:
    objects: List[DetectedObject]
    latency_ms: float


class BaseDetector:
    def detect(self, frame: np.ndarray) -> DetectionResult:
        raise NotImplementedError


class MockDetector(BaseDetector):
    """Fallback detector that estimates dominant colors as pseudo-objects."""

    def detect(self, frame: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        resized = cv2.resize(frame, (64, 64))
        rgb_mean = resized.mean(axis=(0, 1))
        labels = ("blue", "green", "red")
        max_idx = int(np.argmax(rgb_mean))
        top_label = labels[max_idx]
        confidence = float(rgb_mean[max_idx] / 255.0)
        latency_ms = (time.perf_counter() - start) * 1000
        return DetectionResult(
            objects=[
                DetectedObject(
                    label=f"dominant-{top_label}",
                    confidence=confidence,
                    bbox=(0.0, 0.0, 1.0, 1.0),
                )
            ],
            latency_ms=latency_ms,
        )


class HailoDetector(BaseDetector):
    """Thin wrapper around the Hailo-8 runtime.

    The implementation keeps imports out of module scope to allow development
    on machines without the Hailo SDK installed. When running on Raspberry Pi 5
    with a connected Hailo AT module, the detector validates availability and
    uses the compiled HEF to produce detections. The class relies on the
    official `hailo_platform` APIs; if `hailo_model_zoo` is installed alongside
    the HEF/YAML pair, post-processing will produce meaningful bounding boxes.
    """

    def __init__(self, hef_path: Path, score_threshold: float = 0.25):
        self.hef_path = hef_path
        self.score_threshold = score_threshold
        self._ensure_paths()

    def _ensure_paths(self) -> None:
        if not self.hef_path.exists():
            raise FileNotFoundError(f"HEF file not found: {self.hef_path}")

    @staticmethod
    def _hailo_available() -> bool:
        return importlib.util.find_spec("hailo_platform") is not None

    def detect(self, frame: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        if not self._hailo_available():
            raise ImportError(
                "hailo_platform is not installed. Install the Hailo SDK and rerun "
                "without the --demo flag to enable NPU inference."
            )

        # Lazy import to keep development environment light-weight.
        from hailo_platform import HEF, Device  # type: ignore

        with Device() as device:
            hef = HEF(str(self.hef_path))
            configure_params = device.create_configure_params(hef)
            network_groups = device.configure(hef, configure_params)
            network_group = network_groups[0]
            input_vstream_info = network_group.get_input_vstream_infos()[0]
            output_vstream_info = network_group.get_output_vstream_infos()[0]

            # Convert the frame to planar RGB expected by most HEF demos.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            normalized = rgb_frame.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(normalized.transpose(2, 0, 1), axis=0)

            with device.create_infer_streams(network_group, [input_vstream_info], [output_vstream_info]) as streams:
                input_stream = streams.get_input_streams()[0]
                output_stream = streams.get_output_streams()[0]
                input_stream.write(input_tensor)
                raw_output = output_stream.read()

        detections = self._postprocess_output(raw_output)
        latency_ms = (time.perf_counter() - start) * 1000
        filtered = [
            det for det in detections if det.confidence >= self.score_threshold
        ]
        LOGGER.debug("Hailo detected %d objects (%.1f ms)", len(filtered), latency_ms)
        return DetectionResult(objects=filtered, latency_ms=latency_ms)

    def _postprocess_output(self, output_tensors: Sequence[np.ndarray]) -> List[DetectedObject]:
        """Basic post-processing for single-output detection HEF files.

        For production workloads, prefer the post-processing utilities from
        `hailo_model_zoo` that match your HEF and YAML metadata. This helper
        keeps a minimal fallback that treats the output as YOLO-style tensors
        where the last dimension contains [x, y, w, h, objectness, class_scores...].
        """
        if len(output_tensors) != 1:
            return []

        tensor = np.array(output_tensors[0])
        if tensor.ndim != 3:
            return []

        detections: List[DetectedObject] = []
        num_anchors = tensor.shape[1]
        for anchor_idx in range(num_anchors):
            prediction = tensor[0, anchor_idx]
            if prediction.size < 6:
                continue
            x_center, y_center, width, height, obj_score = prediction[:5]
            class_scores = prediction[5:]
            class_id = int(np.argmax(class_scores))
            confidence = float(obj_score * class_scores[class_id])
            if confidence < self.score_threshold:
                continue
            x0 = max(float(x_center - width / 2.0), 0.0)
            y0 = max(float(y_center - height / 2.0), 0.0)
            x1 = min(float(x_center + width / 2.0), 1.0)
            y1 = min(float(y_center + height / 2.0), 1.0)
            detections.append(
                DetectedObject(
                    label=f"class-{class_id}",
                    confidence=confidence,
                    bbox=(x0, y0, x1, y1),
                )
            )

        return detections
