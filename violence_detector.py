"""Video windowing and inference for an RWF-2000 violence classifier."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torchvision.models.video import r3d_18

MODEL_ID = "Abdullahtahir11/rwf2000-violence-detection-3dresnet"
CHECKPOINT_NAME = "rwf2000_best_model.pth"
NUM_FRAMES = 16
FRAME_SIZE = 224
WINDOW_SECONDS = 2.5
STRIDE_SECONDS = 1.0


class DetectorError(RuntimeError):
    """An actionable problem while loading a model or analyzing a video."""


def _checkpoint_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise DetectorError("The downloaded model checkpoint has an unknown format.")

    state = {}
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            state[key.removeprefix("module.")] = value
    if not state:
        raise DetectorError("The model checkpoint does not contain usable weights.")
    return state


def load_detector() -> tuple[nn.Module, torch.device]:
    """Download the public checkpoint on first use and load it on the best device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint_path = hf_hub_download(repo_id=MODEL_ID, filename=CHECKPOINT_NAME)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        model = r3d_18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(_checkpoint_state_dict(checkpoint), strict=True)
        model.to(device).eval()
        return model, device
    except DetectorError:
        raise
    except Exception as exc:
        raise DetectorError(
            "Could not load the pretrained model. Check your internet connection "
            "and installed PyTorch/torchvision versions, then try again."
        ) from exc


def _video_windows(duration: float) -> list[float]:
    if duration <= WINDOW_SECONDS:
        return [0.0]
    starts = list(np.arange(0.0, duration - WINDOW_SECONDS + 1e-6, STRIDE_SECONDS))
    last_start = duration - WINDOW_SECONDS
    if not starts or last_start - starts[-1] > 0.05:
        starts.append(last_start)
    return starts


def _read_window(
    capture: cv2.VideoCapture,
    start_seconds: float,
    fps: float,
    frame_count: int,
) -> torch.Tensor:
    indices = np.linspace(0, max(frame_count - 1, 0), NUM_FRAMES).round().astype(int)
    frames = []
    for frame_offset in indices:
        timestamp_ms = start_seconds * 1000 + frame_offset * 1000 / fps
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        ok, frame = capture.read()
        if not ok or frame is None:
            if frames:
                frame = frames[-1]
            else:
                raise DetectorError(
                    "Could not read frames from this video. Try another video format."
                )
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(
                frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA
            )
        frames.append(frame)

    array = np.stack(frames).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype).view(3, 1, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype).view(3, 1, 1, 1)
    return (tensor - mean) / std


def _format_time(seconds: float) -> str:
    total_milliseconds = round(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}.{milliseconds:03}"


def _merge_segments(
    windows: list[tuple[float, float, float]], threshold: float
) -> list[dict[str, float | str]]:
    flagged = [window for window in windows if window[2] >= threshold]
    merged: list[list[float]] = []
    for start, end, confidence in flagged:
        if merged and start <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] = max(merged[-1][2], confidence)
        else:
            merged.append([start, end, confidence])
    return [
        {
            "start": start,
            "end": end,
            "confidence": confidence,
            "start_label": _format_time(start),
            "end_label": _format_time(end),
        }
        for start, end, confidence in merged
    ]


def analyze_video(
    video_path: str | Path,
    detector: tuple[nn.Module, torch.device],
    threshold: float = 0.5,
    progress_callback: Callable[[float], None] | None = None,
) -> dict[str, object]:
    """Score overlapping 2.5-second windows and merge adjacent flagged windows."""
    model, device = detector
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise DetectorError("Could not open this video. Try MP4, MOV, AVI, or MKV.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0 or frame_count <= 0:
        capture.release()
        raise DetectorError("Could not read the video's duration or frame rate.")

    duration = frame_count / fps
    starts = _video_windows(duration)
    frames_per_window = max(1, min(round(WINDOW_SECONDS * fps), frame_count))
    windows: list[tuple[float, float, float]] = []
    try:
        with torch.inference_mode():
            for index, start in enumerate(starts):
                clip = _read_window(capture, start, fps, frames_per_window)
                probabilities = torch.softmax(model(clip.unsqueeze(0).to(device)), dim=1)
                confidence = float(probabilities[0, 1].item())
                windows.append((start, min(start + WINDOW_SECONDS, duration), confidence))
                if progress_callback:
                    progress_callback((index + 1) / len(starts))
    finally:
        capture.release()

    segments = _merge_segments(windows, threshold)
    return {
        "duration": duration,
        "max_confidence": max((item[2] for item in windows), default=0.0),
        "segments": segments,
    }
