"""Rank Video 4's 15-second rPPG HR windows and help localize the disagreement.

Run this script from the project directory that contains the `results` folder:
    python diagnose_video4_windows.py

It searches the newest Video 4 result folder containing three_signal_comparison.csv,
prints the per-window comparison and saves video4_window_diagnosis.csv beside it.
"""
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
INPUT_NAME = "three_signal_comparison.csv"
OUTPUT_NAME = "video4_window_diagnosis.csv"


def first_existing_column(df, candidates, required=True):
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}\n"
            f"Columns present: {list(df.columns)}"
        )
    return None


def main():
    if not RESULTS_DIR.is_dir():
        print(f"ERROR: '{RESULTS_DIR.resolve()}' was not found.")
        print("Run this script from your project folder, where the results folder exists.")
        sys.exit(1)

    all_csvs = list(RESULTS_DIR.rglob(INPUT_NAME))
    # Match folders/names such as vid_4_20261015_..., vid4_..., or video_4_...
    video4_csvs = [
        p for p in all_csvs
        if re.search(r"(?:vid|video)[_-]?4(?:[_-]|$)", str(p).lower())
    ]

    if not video4_csvs:
        print(f"ERROR: No '{INPUT_NAME}' found under '{RESULTS_DIR.resolve()}' for Video 4.")
        print("CSV files found under results:")
        for p in all_csvs[:30]:
            print("  ", p)
        print("Check that the Video 4 pipeline completed Step 9 and saved its comparison CSV.")
        sys.exit(1)

    # Latest modified comparison file is used so reruns are handled automatically.
    input_path = max(video4_csvs, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(input_path)
    if df.empty:
        print(f"ERROR: The file is empty: {input_path}")
        sys.exit(1)

    try:
        start_col = first_existing_column(df, ["Start_sec", "Window_Start_sec", "Window_Start_s", "Start_s"], required=False)
        end_col = first_existing_column(df, ["End_sec", "Window_End_sec", "Window_End_s", "End_s"], required=False)
        center_col = first_existing_column(df, ["Center_sec", "Window_Center_sec", "Center_s"], required=False)
        camera_col = first_existing_column(df, ["Camera_HR_Interpolated", "Camera_rPPG_HR", "Camera_HR_BPM", "HR_Interpolated_BPM"])
        ppg_col = first_existing_column(df, ["Reference_PPG_HR_Interpolated", "Reference_PPG_HR", "Reference_PPG_HR_BPM"], required=False)
        gt_mean_col = first_existing_column(df, ["GT_Mean_HR", "Supplied_HR_Mean", "GT_Mean_BPM", "GroundTruth_Mean_HR"])
        gt_median_col = first_existing_column(df, ["GT_Median_HR", "Supplied_HR_Median", "GT_Median_BPM"], required=False)
    except KeyError as exc:
        print(f"ERROR: {exc}")
        print(f"Input file: {input_path}")
        sys.exit(1)

    out = pd.DataFrame()
    if start_col:
        out["Start_sec"] = pd.to_numeric(df[start_col], errors="coerce")
    if end_col:
        out["End_sec"] = pd.to_numeric(df[end_col], errors="coerce")
    if center_col:
        out["Center_sec"] = pd.to_numeric(df[center_col], errors="coerce")
    elif "Start_sec" in out and "End_sec" in out:
        out["Center_sec"] = (out["Start_sec"] + out["End_sec"]) / 2.0

    out["Camera_rPPG_HR_BPM"] = pd.to_numeric(df[camera_col], errors="coerce")
    out["Supplied_GT_Mean_HR_BPM"] = pd.to_numeric(df[gt_mean_col], errors="coerce")
    if gt_median_col:
        out["Supplied_GT_Median_HR_BPM"] = pd.to_numeric(df[gt_median_col], errors="coerce")
    if ppg_col:
        out["Reference_PPG_HR_BPM"] = pd.to_numeric(df[ppg_col], errors="coerce")
        out["Camera_vs_PPG_Abs_Error_BPM"] = (
            out["Camera_rPPG_HR_BPM"] - out["Reference_PPG_HR_BPM"]
        ).abs()
        out["PPG_vs_GT_Abs_Error_BPM"] = (
            out["Reference_PPG_HR_BPM"] - out["Supplied_GT_Mean_HR_BPM"]
        ).abs()
    out["Camera_vs_GT_Signed_Error_BPM"] = (
        out["Camera_rPPG_HR_BPM"] - out["Supplied_GT_Mean_HR_BPM"]
    )
    out["Camera_vs_GT_Abs_Error_BPM"] = out["Camera_vs_GT_Signed_Error_BPM"].abs()
    out["Within_5_BPM"] = out["Camera_vs_GT_Abs_Error_BPM"] <= 5.0

    # Brief interpretation based on the three estimates; this is a diagnostic clue, not proof.
    if ppg_col:
        def classify(row):
            c_gt = row["Camera_vs_GT_Abs_Error_BPM"]
            c_ppg = row["Camera_vs_PPG_Abs_Error_BPM"]
            p_gt = row["PPG_vs_GT_Abs_Error_BPM"]
            if not np.isfinite(c_gt) or not np.isfinite(c_ppg) or not np.isfinite(p_gt):
                return "Insufficient values"
            if c_gt <= 5 and c_ppg <= 5 and p_gt <= 5:
                return "All three estimates agree closely"
            if c_ppg <= 3 and c_gt > 5 and p_gt > 5:
                return "Camera and PPG agree; supplied HR differs"
            if c_ppg > 5 and p_gt <= 5 and c_gt > 5:
                return "Camera rPPG differs; reference PPG is closer to GT"
            if c_ppg > 5 and p_gt > 5:
                return "Three-way disagreement; inspect spectra/time alignment"
            if c_gt <= 5:
                return "Camera close to GT; other estimate differs"
            return "Mixed pattern; inspect this window"
        out["Diagnostic_Clue"] = out.apply(classify, axis=1)

    out = out.dropna(subset=["Camera_rPPG_HR_BPM", "Supplied_GT_Mean_HR_BPM"])
    out = out.sort_values("Camera_vs_GT_Abs_Error_BPM", ascending=False).reset_index(drop=True)
    out.insert(0, "Error_Rank_Worst_First", np.arange(1, len(out) + 1))
    out_path = input_path.parent / OUTPUT_NAME
    out.to_csv(out_path, index=False, float_format="%.3f")

    print("\nVIDEO 4 WINDOW DIAGNOSIS")
    print("Input:", input_path.resolve())
    print("Rows analyzed:", len(out))
    print("Saved:", out_path.resolve())
    print("\nWorst-error windows first (HR values in BPM):")
    print(out.round(3).to_string(index=False))
    print("\nHow to read the diagnostic clue:")
    print("- Camera and PPG agree but both differ from supplied HR: check GT HR-column behavior/time alignment first.")
    print("- Reference PPG is close to supplied HR but camera rPPG differs: inspect camera ROI/signal spectrum in that window.")
    print("- All three disagree: the window is ambiguous and needs closer signal/time inspection.")
    print("These are clues, not definitive diagnoses; the estimates are dominant spectral frequencies.")


if __name__ == "__main__":
    main()
