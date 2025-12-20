from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CaptioningConfig:
    """Configuration bundle for the vision captioning pipeline."""

    hef_path: Optional[Path]
    prompt: str
    llm_enabled: bool
    camera_index: int
    frame_width: int
    frame_height: int
    interval_seconds: float
    speak: bool
    demo_mode: bool
    openai_model: str = "gpt-4o-mini"

    @staticmethod
    def from_args(args: argparse.Namespace) -> "CaptioningConfig":
        return CaptioningConfig(
            hef_path=Path(args.hef_path) if args.hef_path else None,
            prompt=args.prompt,
            llm_enabled=not args.no_llm,
            camera_index=args.camera_index,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            interval_seconds=args.interval,
            speak=args.speak,
            demo_mode=args.demo,
            openai_model=args.model,
        )
