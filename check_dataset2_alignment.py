#!/usr/bin/env python3
"""Compare Dataset 2 video duration with the parsed three-line ground truth.

This is a diagnostic only: it does not resample, shift, or modify the signals.

Usage:
  python check_dataset2_alignment.py "D:\\RPPG\\video.avi" "D:\\RPPG\\ground_truth_parsed.csv"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV is required. Install it with: pip install opencv-python")
    sys.exit(2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check video/ground-truth duration and Dataset 2 timestamp irregularity."
    )
    parser.add_argument("video", help="Path to the video paired with this ground truth")
    parser.add_argument(
        "ground_truth_csv",
        nargs="?",
        default="ground_truth_parsed.csv",
        help="CSV produced by parse_dataset2_ground_truth.py (default: ground_truth_parsed.csv)",
    )
    args = parser.parse_args()

    video_path = Path(args.video).expanduser()
    gt_path = Path(args.ground_truth_csv).expanduser()
    if not video_path.is_file():
        print(f"ERROR: Video not found: {video_path}")
        return 2
    if not gt_path.is_file():
        print(f"ERROR: Parsed ground-truth CSV not found: {gt_path}")
        print("Run parse_dataset2_ground_truth.py on the matching ground_truth.txt first.")
        return 2

    try:
        gt = pd.read_csv(gt_path)
    except Exception as exc:
        print(f"ERROR: Could not read CSV: {exc}")
        return 2

    required = {"Timestamp_sec", "HR_BPM", "PPG"}
    if not required.issubset(gt.columns):
        print("ERROR: Parsed CSV must contain columns: Timestamp_sec, HR_BPM, PPG")
        print(f"Found columns: {list(gt.columns)}")
        return 2

    for col in required:
        gt[col] = pd.to_numeric(gt[col], errors="coerce")
    if gt[list(required)].isna().any().any():
        print("ERROR: Parsed CSV contains missing or non-numeric required values.")
        return 2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: OpenCV could not open video: {video_path}")
        return 2
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    video_duration = frame_count / fps if fps > 0 and frame_count > 0 else float("nan")

    t = gt["Timestamp_sec"].to_numpy(dtype=float)
    hr = gt["HR_BPM"].to_numpy(dtype=float)
    ppg = gt["PPG"].to_numpy(dtype=float)
    dt = np.diff(t)
    positive_dt = dt[dt > 0]
    duplicate_steps = int(np.sum(dt == 0))
    backward_steps = int(np.sum(dt < 0))

    if len(positive_dt):
        med_dt = float(np.median(positive_dt))
        mean_dt = float(np.mean(positive_dt))
        p95_dt = float(np.percentile(positive_dt, 95))
        max_dt = float(np.max(positive_dt))
        min_dt = float(np.min(positive_dt))
        med_rate = 1.0 / med_dt if med_dt > 0 else float("nan")
        mean_rate = 1.0 / mean_dt if mean_dt > 0 else float("nan")
        gap_threshold = 1.5 * med_dt
        large_gaps = int(np.sum(positive_dt > gap_threshold))
    else:
        med_dt = mean_dt = p95_dt = max_dt = min_dt = med_rate = mean_rate = gap_threshold = float("nan")
        large_gaps = 0

    gt_duration = float(np.max(t) - np.min(t)) if len(t) else float("nan")
    duration_delta = gt_duration - video_duration
    count_rate = (len(t) - 1) / gt_duration if gt_duration > 0 else float("nan")

    print("DATASET 2 VIDEO / GROUND-TRUTH ALIGNMENT CHECK")
    print(f"Video: {video_path.resolve()}")
    print(f"Parsed ground truth: {gt_path.resolve()}")
    print("\nVIDEO METADATA")
    print(f"FPS: {fps:.6f}")
    print(f"Frames: {frame_count}")
    print(f"Resolution: {width} x {height}")
    print(f"Video duration (frames/FPS): {video_duration:.6f} s")
    print("\nGROUND-TRUTH METADATA")
    print(f"Rows: {len(gt)}")
    print(f"Timestamp start: {np.min(t):.6f} s")
    print(f"Timestamp end:   {np.max(t):.6f} s")
    print(f"Timestamp duration: {gt_duration:.6f} s")
    print(f"Ground truth minus video duration: {duration_delta:+.6f} s")
    print(f"Duration difference (absolute): {abs(duration_delta):.6f} s")
    print(f"Approx. row rate from count/duration: {count_rate:.3f} Hz")
    print("\nTIMESTAMP QUALITY")
    print(f"Duplicate steps: {duplicate_steps}")
    print(f"Backward steps: {backward_steps}")
    print(f"Median positive interval: {med_dt:.6f} s")
    print(f"Mean positive interval:   {mean_dt:.6f} s")
    print(f"Minimum positive interval: {min_dt:.6f} s")
    print(f"95th percentile interval: {p95_dt:.6f} s")
    print(f"Maximum positive interval: {max_dt:.6f} s")
    print(f"Rate from median interval: {med_rate:.3f} Hz")
    print(f"Rate from mean interval:   {mean_rate:.3f} Hz")
    print(f"Long gaps (> 1.5 x median interval): {large_gaps}")
    print(f"Gap threshold: {gap_threshold:.6f} s")
    print("\nHR/PPG SANITY CHECK")
    print(f"HR min / median / max: {np.min(hr):.3f} / {np.median(hr):.3f} / {np.max(hr):.3f} BPM")
    print(f"PPG min / median / max: {np.min(ppg):.6f} / {np.median(ppg):.6f} / {np.max(ppg):.6f}")

    print("\nINTERPRETATION")
    if backward_steps:
        print("- Timestamps run backward. Do not evaluate until the source ordering is understood.")
    if duplicate_steps:
        print("- Duplicate timestamps exist. Preserve the original file; define a duplicate-handling policy before interpolation.")
    if large_gaps:
        print("- Timestamp gaps are irregular. FFT requires a uniformly sampled signal, so do not apply FFT directly to these raw PPG values as if sampling were uniform.")
    if np.isfinite(duration_delta) and abs(duration_delta) > 1.0:
        print("- Ground-truth and video durations differ by more than 1 second. Check that this is the matching file and whether timestamps include offsets or missing segments.")
    elif np.isfinite(duration_delta):
        print("- Durations are within 1 second, but this alone does not prove exact synchronization.")
    print("- This script is diagnostic only; no resampling, time shifting, or source-data modification was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
