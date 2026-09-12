"""Stage 1 — ingest: video decoding, telemetry parsing, adaptive frame selection."""

from src.ingest.frame_selector import FrameSelection, select_frames
from src.ingest.telemetry import TelemetryTable, load_telemetry
from src.ingest.video_reader import Frame, VideoMetadata, VideoReader

__all__ = [
    "Frame",
    "VideoMetadata",
    "VideoReader",
    "TelemetryTable",
    "load_telemetry",
    "FrameSelection",
    "select_frames",
]
