from datetime import datetime
import os
from pathlib import Path
import shutil
import sys

import cv2
import pandas as pd

from step1_ROI_txt import extract_roi_rgb
from step2_average import average_rgb
from step3_pos_signal_window import calculate_pos
from step4_bandpass_filter import filter_rppg
from step5_fft_hr import estimate_heart_rate
from step6_ground_truth import compare_heart_rate
from step7_windowed_hr import evaluate_windowed_hr
from step8_10sec_sliding_window import evaluate_sliding_hr
from step9_compare_3_signal import compare_three_hr_signals


PROJECT_DIR = Path(__file__).resolve().parent

# Updated input directories
VIDEO_DIR = PROJECT_DIR / "video"
GT_DIR = PROJECT_DIR / "data_set" / "data_xmp"

EXPECTED_ARTIFACTS = [
    "all_roi_rgb.csv",
    "average_rgb.csv",
    "pos_signal_windowed.csv",
    "filtered_rppg.csv",
]


def resolve_input(raw_path: str, label: str) -> Path:
    cleaned = raw_path.strip().strip('"').strip("'")

    if not cleaned:
        raise ValueError(f"No {label} path was provided.")

    path = Path(cleaned).expanduser()

    if not path.is_absolute():

        if label == "Video":
            base_dir = VIDEO_DIR

        elif label == "Ground-truth":
            base_dir = GT_DIR

        else:
            base_dir = PROJECT_DIR

        # First try the corresponding input folder
        candidate = base_dir / path

        if candidate.is_file():
            path = candidate
        else:
            # Also support project-relative paths
            path = PROJECT_DIR / path

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"{label} file not found: {path}"
        )

    return path



# def resolve_input(raw_path: str, label: str) -> Path:
#     cleaned = raw_path.strip().strip('"').strip("'")
#     if not cleaned:
#         raise ValueError(f"No {label} path was provided.")

#     path = Path(cleaned).expanduser()
#     if not path.is_absolute():
#         path = PROJECT_DIR / path

#     path = path.resolve()
#     if not path.is_file():
#         raise FileNotFoundError(f"{label} file not found: {path}")
#     return path


def ensure_artifact(filename: str, output_dir: Path) -> Path:

    target = output_dir / filename
    if target.is_file():
        return target

    source = PROJECT_DIR / filename
    if source.is_file():
        shutil.copy2(source, target)
        return target

    raise FileNotFoundError(
        f"Expected output '{filename}' was not created in either "
        f"'{output_dir}' or '{PROJECT_DIR}'. Check the corresponding stage function."
    )


def check_frame_rows(filename: str, expected_rows: int, output_dir: Path) -> None:
    path = ensure_artifact(filename, output_dir)
    data = pd.read_csv(path)
    if len(data) != expected_rows:
        raise ValueError(
            f"Frame/sample count mismatch in {filename}: expected "
            f"{expected_rows} rows for the video, found {len(data)}. "
        )
    print(f"Verified {filename}: {len(data)} rows")


def main(video_input: str | None = None, ground_truth_input: str | None = None) -> Path:
    if video_input is None:
        video_input = input(
            "Enter video path (e.g. vid_5.avi, or full path): "
        )
    if ground_truth_input is None:
        ground_truth_input = input(
            "Enter matching ground-truth path (e.g. gtdump_5.xmp): "
        )

    video_path = resolve_input(video_input, "Video")
    gt_path = resolve_input(ground_truth_input, "Ground-truth")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or frame_count <= 0:
        raise ValueError(
            f"Invalid video metadata: FPS={fps}, frame count={frame_count}"
        )

    duration = frame_count / fps

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = PROJECT_DIR / "results"
    output_dir = results_root / f"{video_path.stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    original_cwd = Path.cwd()
    our_hr = None

    try:
        os.chdir(output_dir)

        print("\n========== RPPG PIPELINE RUN ==========")
        print("Video:", video_path)
        print("Ground truth:", gt_path)
        print("Results folder:", output_dir)
        print("FPS:", fps)
        print("Frame count:", frame_count)
        print("Resolution:", f"{width} x {height}")
        print("Duration:", f"{duration:.3f} seconds")

        print("\n[1/9] Extracting ROI RGB values...")
        extract_roi_rgb(str(video_path))
        ensure_artifact("all_roi_rgb.csv", output_dir)
        check_frame_rows("all_roi_rgb.csv", frame_count, output_dir)

        print("\n[2/9] Averaging RGB values...")
        average_rgb("all_roi_rgb.csv")
        check_frame_rows("average_rgb.csv", frame_count, output_dir)

        print("\n[3/9] Calculating windowed POS signal...")
        calculate_pos("average_rgb.csv",fps)
        check_frame_rows("pos_signal_windowed.csv", frame_count, output_dir)

        print("\n[4/9] Filtering POS signal...")
        filter_rppg("pos_signal_windowed.csv",fps)
        check_frame_rows("filtered_rppg.csv", frame_count, output_dir)

        print("\n[5/9] Estimating whole-recording HR...")
        our_hr = estimate_heart_rate("filtered_rppg.csv",fps)

        print("\n[6/9] Comparing whole-recording HR with ground truth...")
        compare_heart_rate(str(gt_path), our_hr)

        print("\n[7/9] Evaluating complete non-overlapping 10-second windows...")
        evaluate_windowed_hr("filtered_rppg.csv", str(gt_path),fps)

        print("\n[8/9] Evaluating sliding HR windows...")
        evaluate_sliding_hr("filtered_rppg.csv", str(gt_path),fps)

        print("\n[9/9] Comparing camera rPPG, reference PPG, and supplied HR...")
        compare_three_hr_signals("filtered_rppg.csv", str(gt_path),fps)

        summary_path = output_dir / "run_summary.txt"
        summary_path.write_text(
            "rPPG pipeline run summary\n"
            f"Video: {video_path}\n"
            f"Ground truth: {gt_path}\n"
            f"FPS: {fps}\n"
            f"Frame count: {frame_count}\n"
            f"Width: {width}\n"
            f"Height: {height}\n"
            f"Duration seconds: {duration}\n"
            f"Whole-recording estimated HR BPM: {our_hr}\n",
            encoding="utf-8"
        )

        print("\n========== PIPELINE COMPLETE ==========")
        print("Whole-recording estimated HR:", our_hr, "BPM")
        print("Results saved in:", output_dir)
        print("=======================================")

        return output_dir

    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":

    if len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main()
