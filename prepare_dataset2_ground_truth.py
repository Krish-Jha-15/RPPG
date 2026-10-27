#!/usr/bin/env python3
"""Prepare Dataset 2 ground_truth.txt for evaluation against a video.

Expected non-empty lines in ground_truth.txt:
  1. PPG samples
  2. HR samples in BPM
  3. timestamps in seconds

This script does NOT modify the source text file or any rPPG pipeline output.
It combines duplicate timestamps by averaging PPG and HR at that timestamp,
then linearly interpolates PPG and HR onto the video's frame-time grid. Only
frame times inside the common video/ground-truth time range are retained.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_numeric_line(line: str, line_name: str) -> np.ndarray:
    tokens = line.split()
    if not tokens:
        raise ValueError(f"{line_name} is empty")
    try:
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{line_name} contains a non-numeric token: {exc}") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{line_name} contains NaN or infinite values")
    return values


def read_ground_truth(txt_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        text = txt_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = txt_path.read_text(encoding="latin-1")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(
            f"Expected at least 3 non-empty lines (PPG, HR, timestamps); found {len(lines)}"
        )
    if len(lines) > 3:
        # The format description specifies three signal lines. Avoid silently
        # interpreting unexpected extra content as part of the data.
        extra = [line for line in lines[3:] if line and not line.startswith("#")]
        if extra:
            raise ValueError(
                f"Expected 3 non-empty signal lines, found {len(lines)}. "
                "Check whether the file includes headers or additional data lines."
            )
    ppg = parse_numeric_line(lines[0], "Line 1 (PPG)")
    hr = parse_numeric_line(lines[1], "Line 2 (HR)")
    timestamps = parse_numeric_line(lines[2], "Line 3 (timestamps)")
    if not (len(ppg) == len(hr) == len(timestamps)):
        raise ValueError(
            "Sample counts differ: "
            f"PPG={len(ppg)}, HR={len(hr)}, timestamps={len(timestamps)}"
        )
    if len(timestamps) < 2:
        raise ValueError("At least two timestamp samples are required")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("Timestamps go backwards; resolve ordering before resampling")
    return ppg, hr, timestamps


def video_metadata(video_path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not np.isfinite(fps) or fps <= 0 or frame_count < 2:
        raise ValueError(f"Invalid video metadata: FPS={fps}, frames={frame_count}")
    return fps, frame_count, width, height


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resample Dataset 2 PPG/HR onto matching video frame times for evaluation."
    )
    parser.add_argument("video", help="Path to matching video file, e.g. D:\\RPPG\\vid_8.avi")
    parser.add_argument("ground_truth_txt", help="Path to Dataset 2 ground_truth.txt")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder (default: <video folder>\\dataset2_prepared)",
    )
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    txt_path = Path(args.ground_truth_txt).expanduser().resolve()
    if not video_path.is_file():
        print(f"ERROR: Video file not found: {video_path}", file=sys.stderr)
        return 2
    if not txt_path.is_file():
        print(f"ERROR: Ground-truth file not found: {txt_path}", file=sys.stderr)
        return 2

    try:
        ppg, hr, timestamps = read_ground_truth(txt_path)
        fps, frame_count, width, height = video_metadata(video_path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    deltas = np.diff(timestamps)
    positive = deltas[deltas > 0]
    if positive.size == 0:
        print("ERROR: No positive timestamp intervals; cannot resample.", file=sys.stderr)
        return 2

    duplicate_rows = int(np.sum(deltas == 0))
    median_dt = float(np.median(positive))
    gap_threshold = 1.5 * median_dt
    long_gaps = int(np.sum(positive > gap_threshold))

    # Collapse repeated timestamps. Mean-aggregation is explicit and only used
    # for duplicate time points so np.interp gets strictly increasing x values.
    source = pd.DataFrame({"Timestamp_sec": timestamps, "HR_BPM": hr, "PPG": ppg})
    collapsed = (
        source.groupby("Timestamp_sec", as_index=False, sort=True)
        .agg(HR_BPM=("HR_BPM", "mean"), PPG=("PPG", "mean"))
    )
    t_src = collapsed["Timestamp_sec"].to_numpy(dtype=float)
    hr_src = collapsed["HR_BPM"].to_numpy(dtype=float)
    ppg_src = collapsed["PPG"].to_numpy(dtype=float)

    video_frame_times = np.arange(frame_count, dtype=float) / fps
    video_last_frame_time = float(video_frame_times[-1])
    common_start = max(float(t_src[0]), float(video_frame_times[0]))
    common_end = min(float(t_src[-1]), video_last_frame_time)
    keep = (video_frame_times >= common_start - 1e-12) & (
        video_frame_times <= common_end + 1e-12
    )
    target_times = video_frame_times[keep]
    if len(target_times) < 2:
        print("ERROR: Fewer than two frame timestamps fall inside the common time range.", file=sys.stderr)
        return 2

    ppg_uniform = np.interp(target_times, t_src, ppg_src)
    hr_uniform = np.interp(target_times, t_src, hr_src)
    frame_numbers = np.arange(frame_count, dtype=int)[keep] + 1  # one-based frame numbers
    out = pd.DataFrame(
        {
            "Frame": frame_numbers,
            "Timestamp_sec": target_times,
            "HR_BPM": hr_uniform,
            "PPG": ppg_uniform,
        }
    )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else video_path.parent / "dataset2_prepared"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{video_path.stem}_ground_truth_resampled.csv"
    report_path = output_dir / f"{video_path.stem}_ground_truth_resampling_report.txt"
    out.to_csv(output_csv, index=False, float_format="%.9f")

    summary = [
        "DATASET 2 GROUND-TRUTH RESAMPLING REPORT",
        f"Video: {video_path}",
        f"Original ground truth: {txt_path}",
        f"Output CSV: {output_csv}",
        "",
        "VIDEO METADATA",
        f"FPS: {fps:.9f}",
        f"Frame count: {frame_count}",
        f"Resolution: {width} x {height}",
        f"Last frame timestamp: {video_last_frame_time:.9f} s",
        f"Frame-count/FPS duration: {frame_count / fps:.9f} s",
        "",
        "SOURCE GROUND TRUTH",
        f"PPG samples: {len(ppg)}",
        f"HR samples: {len(hr)}",
        f"Timestamp samples: {len(timestamps)}",
        f"Source timestamp range: {timestamps[0]:.9f} to {timestamps[-1]:.9f} s",
        f"Source duplicate timestamp steps: {duplicate_rows}",
        f"Unique timestamps after duplicate averaging: {len(t_src)}",
        f"Median positive timestamp interval: {median_dt:.9f} s",
        f"Approximate median-interval rate (descriptive only): {1.0 / median_dt:.6f} Hz",
        f"Large positive intervals (> 1.5 x median): {long_gaps}",
        f"Large-gap threshold: {gap_threshold:.9f} s",
        "",
        "COMMON TIME RANGE AND RESAMPLING",
        f"Common start: {common_start:.9f} s",
        f"Common end: {common_end:.9f} s",
        f"Common duration: {common_end - common_start:.9f} s",
        f"Output rows: {len(out)} (video-frame times only within common duration)",
        f"Output time step: {1.0 / fps:.9f} s (uniform video frame grid)",
        "Duplicate policy: average HR and PPG values at identical timestamps.",
        "Interpolation: linear interpolation from original timestamps to video frame times.",
        "Original ground_truth.txt was not modified.",
        "Only shared-time samples are output; no extrapolation past the available GT duration.",
        "",
        "FIRST 5 OUTPUT ROWS",
        out.head(5).to_string(index=False),
        "",
        "CAUTION",
        "Detected large timestamp gaps are bridged by linear interpolation. Inspect the source if the PPG spectrum near a gap is important.",
        "Interpolation enables uniform-grid FFT analysis but cannot recover information lost during a gap.",
    ]
    report_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("DATASET 2 GROUND-TRUTH RESAMPLING COMPLETE")
    print(f"Video: {video_path}")
    print(f"Ground truth: {txt_path}")
    print(f"FPS: {fps:.6f}; video frames: {frame_count}; last frame time: {video_last_frame_time:.6f} s")
    print(f"Source GT rows: {len(timestamps)}; duplicate timestamp steps: {duplicate_rows}; unique times: {len(t_src)}")
    print(f"GT time range: {timestamps[0]:.6f} to {timestamps[-1]:.6f} s")
    print(f"Common time range: {common_start:.6f} to {common_end:.6f} s")
    print(f"Uniform output rows: {len(out)}; frame interval: {1.0 / fps:.6f} s")
    print(f"Median positive GT interval: {median_dt:.6f} s; largest interval: {float(np.max(positive)):.6f} s")
    print(f"Large intervals (> {gap_threshold:.6f} s): {long_gaps}")
    if duplicate_rows:
        print("Duplicate policy: average PPG and HR values for identical timestamps.")
    if long_gaps:
        print("WARNING: one or more large timestamp gaps will be bridged by linear interpolation.")
    print(f"Saved CSV: {output_csv}")
    print(f"Saved report: {report_path}")
    print("Note: This prepares GT only; it does not run/change the rPPG pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
