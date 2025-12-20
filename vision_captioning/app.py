from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

from .config import CaptioningConfig
from .detector import HailoDetector, MockDetector
from .llm_client import CaptioningLLM
from .pipeline import VisionCaptioningPipeline
from .tts_client import TTSClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hailo-accelerated vision captioning with TTS.")
    parser.add_argument("--hef-path", type=str, help="Path to HEF file for Hailo inference.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="You are a precise captioning agent. Respond in Korean with a single sentence.",
        help="System prompt for the LLM.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Index of the USB camera.")
    parser.add_argument("--frame-width", type=int, default=640, help="Frame width in pixels.")
    parser.add_argument("--frame-height", type=int, default=480, help="Frame height in pixels.")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between captures.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name.")
    parser.add_argument("--speak", action="store_true", help="Enable TTS playback.")
    parser.add_argument("--demo", action="store_true", help="Use mock detections without Hailo.")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls and log detections.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def build_detector(config: CaptioningConfig):
    if config.demo_mode:
        return MockDetector()
    if not config.hef_path:
        raise ValueError("Hailo mode requires --hef-path pointing to a compiled HEF model.")
    return HailoDetector(hef_path=config.hef_path)


def build_llm_client(config: CaptioningConfig):
    if not config.llm_enabled:
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is required when LLM is enabled.")
    return CaptioningLLM(model=config.openai_model)


def main() -> None:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = CaptioningConfig.from_args(args)

    detector = build_detector(config)
    llm_client = build_llm_client(config)
    tts_client = TTSClient() if config.speak else None

    pipeline = VisionCaptioningPipeline(
        config=config,
        detector=detector,
        llm_client=llm_client,
        tts_client=tts_client,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
