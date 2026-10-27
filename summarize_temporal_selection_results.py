"""Summarize baseline vs temporal spectral-candidate selection results.

Scans existing results folders for per-video temporal-selection CSVs and writes:
  - temporal_selection_video_metrics.csv
  - temporal_selection_all_windows.csv
  - temporal_selection_changed_windows.csv
  - temporal_selection_consolidated_report.txt

By design, Video 4 is excluded. Ground-truth HR is used only to evaluate estimates;
this script does not reselect candidates or alter any pipeline outputs.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TARGET_VIDEOS = [1, 2, 3, 5, 6, 7]  # Video 4 intentionally excluded
REQUIRED_COLUMNS = [
    "Start_sec",
    "End_sec",
    "Dominant_Peak_HR_BPM",
    "Temporally_Selected_HR_BPM",
    "GT_HR_Mean_BPM",
]
OPTIONAL_COLUMNS = ["Selected_Candidate_Rank", "Selected_Power_Ratio"]


def get_latest_csv(results_root: Path, video_num: int) -> Path | None:
    """Find the most recently modified temporal-selection output for one video."""
    filename = f"video{video_num}_temporal_peak_selection.csv"
    candidates = list(results_root.rglob(filename))
    if not candidates:
        return None

    # The output may exist in multiple dated result folders. Prefer the most recently
    # modified CSV, which corresponds to the latest diagnostic run in ordinary use.
    candidates.sort(key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    return candidates[0]


def metric_bundle(pred: pd.Series | np.ndarray, gt: pd.Series | np.ndarray) -> dict[str, float | int]:
    pred_arr = np.asarray(pred, dtype=float)
    gt_arr = np.asarray(gt, dtype=float)
    valid = np.isfinite(pred_arr) & np.isfinite(gt_arr)
    pred_arr = pred_arr[valid]
    gt_arr = gt_arr[valid]

    n = int(len(pred_arr))
    if n == 0:
        return {"Windows": 0, "MAE_BPM": np.nan, "RMSE_BPM": np.nan,
                "Bias_BPM": np.nan, "Within_5_BPM_pct": np.nan}

    err = pred_arr - gt_arr
    abs_err = np.abs(err)
    return {
        "Windows": n,
        "MAE_BPM": float(np.mean(abs_err)),
        "RMSE_BPM": float(np.sqrt(np.mean(err ** 2))),
        "Bias_BPM": float(np.mean(err)),
        "Within_5_BPM_pct": float(np.mean(abs_err <= 5.0) * 100.0),
    }


def read_video_csv(video_num: int, csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: required column(s) missing: {', '.join(missing)}")

    for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Preserve only rows where window boundaries and both prediction methods exist.
    # Missing GT remains visible but cannot contribute to accuracy metrics.
    df = df.dropna(subset=["Start_sec", "End_sec", "Dominant_Peak_HR_BPM",
                           "Temporally_Selected_HR_BPM"]).copy()
    df.insert(0, "Video", video_num)
    df.insert(1, "Source_Result_Folder", str(csv_path.parents[2]))
    df.insert(2, "Source_CSV", str(csv_path))

    valid_gt = np.isfinite(df["GT_HR_Mean_BPM"].to_numpy(dtype=float))
    df["GT_Available"] = valid_gt
    df["Baseline_Signed_Error_BPM"] = df["Dominant_Peak_HR_BPM"] - df["GT_HR_Mean_BPM"]
    df["Baseline_Abs_Error_BPM"] = df["Baseline_Signed_Error_BPM"].abs()
    df["Experimental_Signed_Error_BPM"] = df["Temporally_Selected_HR_BPM"] - df["GT_HR_Mean_BPM"]
    df["Experimental_Abs_Error_BPM"] = df["Experimental_Signed_Error_BPM"].abs()
    df["Abs_Error_Improvement_BPM"] = (
        df["Baseline_Abs_Error_BPM"] - df["Experimental_Abs_Error_BPM"]
    )
    df["Estimate_Change_BPM"] = (
        df["Temporally_Selected_HR_BPM"] - df["Dominant_Peak_HR_BPM"]
    )
    df["Window_Changed"] = df["Estimate_Change_BPM"].abs() > 1e-6
    return df


def build_summary(video_frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    per_video_rows: list[dict[str, object]] = []
    all_valid_rows: list[pd.DataFrame] = []
    all_rows = pd.concat(video_frames, ignore_index=True) if video_frames else pd.DataFrame()

    for video_df in video_frames:
        video_num = int(video_df["Video"].iloc[0])
        valid = video_df[np.isfinite(video_df["GT_HR_Mean_BPM"].to_numpy(dtype=float))].copy()
        if not valid.empty:
            all_valid_rows.append(valid)
        baseline = metric_bundle(valid["Dominant_Peak_HR_BPM"], valid["GT_HR_Mean_BPM"])
        experimental = metric_bundle(valid["Temporally_Selected_HR_BPM"], valid["GT_HR_Mean_BPM"])
        changed = int(video_df["Window_Changed"].sum())

        row: dict[str, object] = {
            "Video": video_num,
            "Result_Folder": str(video_df["Source_Result_Folder"].iloc[0]),
            "Windows_In_File": int(len(video_df)),
            "Windows_Evaluated": int(baseline["Windows"]),
            "Windows_Changed": changed,
            "Baseline_MAE_BPM": baseline["MAE_BPM"],
            "Experimental_MAE_BPM": experimental["MAE_BPM"],
            "MAE_Improvement_BPM": (
                float(baseline["MAE_BPM"] - experimental["MAE_BPM"])
                if np.isfinite(baseline["MAE_BPM"]) and np.isfinite(experimental["MAE_BPM"]) else np.nan
            ),
            "Baseline_RMSE_BPM": baseline["RMSE_BPM"],
            "Experimental_RMSE_BPM": experimental["RMSE_BPM"],
            "RMSE_Improvement_BPM": (
                float(baseline["RMSE_BPM"] - experimental["RMSE_BPM"])
                if np.isfinite(baseline["RMSE_BPM"]) and np.isfinite(experimental["RMSE_BPM"]) else np.nan
            ),
            "Baseline_Bias_BPM": baseline["Bias_BPM"],
            "Experimental_Bias_BPM": experimental["Bias_BPM"],
            "Baseline_Within_5_BPM_pct": baseline["Within_5_BPM_pct"],
            "Experimental_Within_5_BPM_pct": experimental["Within_5_BPM_pct"],
        }
        per_video_rows.append(row)

    summary = pd.DataFrame(per_video_rows).sort_values("Video") if per_video_rows else pd.DataFrame()
    valid_pooled = pd.concat(all_valid_rows, ignore_index=True) if all_valid_rows else pd.DataFrame()
    baseline_pooled = metric_bundle(valid_pooled["Dominant_Peak_HR_BPM"], valid_pooled["GT_HR_Mean_BPM"]) if not valid_pooled.empty else metric_bundle([], [])
    experimental_pooled = metric_bundle(valid_pooled["Temporally_Selected_HR_BPM"], valid_pooled["GT_HR_Mean_BPM"]) if not valid_pooled.empty else metric_bundle([], [])

    # Macro averages weight each video equally; pooled metrics weight each evaluated window equally.
    macro: dict[str, float | int] = {"Videos_Evaluated": int(len(summary))}
    for label, col in [
        ("Macro_Baseline_MAE_BPM", "Baseline_MAE_BPM"),
        ("Macro_Experimental_MAE_BPM", "Experimental_MAE_BPM"),
        ("Macro_Baseline_RMSE_BPM", "Baseline_RMSE_BPM"),
        ("Macro_Experimental_RMSE_BPM", "Experimental_RMSE_BPM"),
        ("Macro_Baseline_Within_5_BPM_pct", "Baseline_Within_5_BPM_pct"),
        ("Macro_Experimental_Within_5_BPM_pct", "Experimental_Within_5_BPM_pct"),
    ]:
        macro[label] = float(summary[col].mean()) if not summary.empty else np.nan

    pooled: dict[str, float | int] = {
        "Windows_Evaluated_Pooled": int(baseline_pooled["Windows"]),
        "Pooled_Baseline_MAE_BPM": baseline_pooled["MAE_BPM"],
        "Pooled_Experimental_MAE_BPM": experimental_pooled["MAE_BPM"],
        "Pooled_MAE_Improvement_BPM": (
            float(baseline_pooled["MAE_BPM"] - experimental_pooled["MAE_BPM"])
            if np.isfinite(baseline_pooled["MAE_BPM"]) and np.isfinite(experimental_pooled["MAE_BPM"]) else np.nan
        ),
        "Pooled_Baseline_RMSE_BPM": baseline_pooled["RMSE_BPM"],
        "Pooled_Experimental_RMSE_BPM": experimental_pooled["RMSE_BPM"],
        "Pooled_RMSE_Improvement_BPM": (
            float(baseline_pooled["RMSE_BPM"] - experimental_pooled["RMSE_BPM"])
            if np.isfinite(baseline_pooled["RMSE_BPM"]) and np.isfinite(experimental_pooled["RMSE_BPM"]) else np.nan
        ),
        "Pooled_Baseline_Within_5_BPM_pct": baseline_pooled["Within_5_BPM_pct"],
        "Pooled_Experimental_Within_5_BPM_pct": experimental_pooled["Within_5_BPM_pct"],
        "Pooled_Baseline_Bias_BPM": baseline_pooled["Bias_BPM"],
        "Pooled_Experimental_Bias_BPM": experimental_pooled["Bias_BPM"],
    }
    overall = {**macro, **pooled}
    return summary, all_rows, overall


def write_text_report(
    report_path: Path,
    summary: pd.DataFrame,
    all_rows: pd.DataFrame,
    overall: dict[str, float | int],
    missing_videos: list[int],
) -> None:
    lines: list[str] = []
    lines.append("TEMPORAL SPECTRAL-CANDIDATE SELECTION: CONSOLIDATED EVALUATION")
    lines.append("Videos included: 1, 2, 3, 5, 6, 7 (Video 4 excluded by request)")
    lines.append("Baseline = dominant spectral peak; experimental = temporally selected candidate.")
    lines.append("Ground-truth HR is used only for evaluation in this report.")
    lines.append("")

    if not summary.empty:
        display_cols = [
            "Video", "Windows_Evaluated", "Windows_Changed",
            "Baseline_MAE_BPM", "Experimental_MAE_BPM", "MAE_Improvement_BPM",
            "Baseline_RMSE_BPM", "Experimental_RMSE_BPM", "RMSE_Improvement_BPM",
            "Baseline_Within_5_BPM_pct", "Experimental_Within_5_BPM_pct",
        ]
        lines.append("PER-VIDEO METRICS")
        lines.append(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        lines.append("")

    lines.append("POOLED WINDOW METRICS (each valid window weighted equally)")
    for key in [
        "Windows_Evaluated_Pooled", "Pooled_Baseline_MAE_BPM", "Pooled_Experimental_MAE_BPM",
        "Pooled_MAE_Improvement_BPM", "Pooled_Baseline_RMSE_BPM", "Pooled_Experimental_RMSE_BPM",
        "Pooled_RMSE_Improvement_BPM", "Pooled_Baseline_Within_5_BPM_pct",
        "Pooled_Experimental_Within_5_BPM_pct", "Pooled_Baseline_Bias_BPM", "Pooled_Experimental_Bias_BPM",
    ]:
        val = overall.get(key, np.nan)
        lines.append(f"{key}: {val:.3f}" if isinstance(val, (int, float, np.number)) and np.isfinite(val) else f"{key}: {val}")

    lines.append("")
    lines.append("MACRO AVERAGES (each video weighted equally)")
    for key in [
        "Videos_Evaluated", "Macro_Baseline_MAE_BPM", "Macro_Experimental_MAE_BPM",
        "Macro_Baseline_RMSE_BPM", "Macro_Experimental_RMSE_BPM",
        "Macro_Baseline_Within_5_BPM_pct", "Macro_Experimental_Within_5_BPM_pct",
    ]:
        val = overall.get(key, np.nan)
        lines.append(f"{key}: {val:.3f}" if isinstance(val, (int, float, np.number)) and np.isfinite(val) else f"{key}: {val}")

    changed = all_rows[all_rows["Window_Changed"]].copy() if not all_rows.empty else pd.DataFrame()
    lines.append("")
    lines.append(f"CHANGED WINDOWS ({len(changed)} total)")
    if changed.empty:
        lines.append("No selected HR estimate changed from the dominant-peak baseline.")
    else:
        cols = ["Video", "Start_sec", "End_sec", "GT_HR_Mean_BPM", "Dominant_Peak_HR_BPM",
                "Temporally_Selected_HR_BPM", "Selected_Candidate_Rank", "Selected_Power_Ratio",
                "Baseline_Abs_Error_BPM", "Experimental_Abs_Error_BPM", "Abs_Error_Improvement_BPM"]
        cols = [c for c in cols if c in changed.columns]
        lines.append(changed[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if missing_videos:
        lines.append("")
        lines.append("WARNING: No temporal-selection CSV was found for video(s): " + ", ".join(map(str, missing_videos)))

    lines.append("")
    lines.append("INTERPRETATION NOTES")
    lines.append("- Positive MAE/RMSE improvement means the experimental result is lower than baseline.")
    lines.append("- Negative improvement means the experiment performed worse.")
    lines.append("- Pooled metrics weight each window equally; macro averages weight each video equally.")
    lines.append("- The 15-second windows overlap with a 5-second hop, so window-level results are correlated.")
    lines.append("- This is a diagnostic comparison, not an independent clinical validation. Keep the baseline pipeline intact until broader validation is complete.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results", help="Results directory (default: ./results)")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: <results-root>/temporal_selection_consolidated)",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    if not results_root.exists() or not results_root.is_dir():
        print(f"ERROR: results directory not found: {results_root}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else results_root / "temporal_selection_consolidated"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    missing: list[int] = []
    errors: list[str] = []

    print("TEMPORAL SELECTION CONSOLIDATED REPORT")
    print(f"Results root: {results_root}")
    print(f"Videos included: {', '.join(map(str, TARGET_VIDEOS))}; Video 4 intentionally excluded")
    print("")

    for video_num in TARGET_VIDEOS:
        csv_path = get_latest_csv(results_root, video_num)
        if csv_path is None:
            missing.append(video_num)
            print(f"WARNING: Video {video_num}: temporal-selection CSV not found")
            continue
        try:
            frame = read_video_csv(video_num, csv_path)
            frames.append(frame)
            valid_gt = int(np.isfinite(frame["GT_HR_Mean_BPM"].to_numpy(dtype=float)).sum())
            print(f"Video {video_num}: {len(frame)} window rows; {valid_gt} with GT | {csv_path}")
        except Exception as exc:
            errors.append(f"Video {video_num}: {exc}")
            print(f"ERROR: Video {video_num}: {exc}", file=sys.stderr)

    if not frames:
        print("ERROR: No usable temporal-selection CSVs were found.", file=sys.stderr)
        return 3

    summary, all_rows, overall = build_summary(frames)
    changed = all_rows[all_rows["Window_Changed"]].copy()

    summary_path = output_dir / "temporal_selection_video_metrics.csv"
    all_rows_path = output_dir / "temporal_selection_all_windows.csv"
    changed_path = output_dir / "temporal_selection_changed_windows.csv"
    report_path = output_dir / "temporal_selection_consolidated_report.txt"

    summary.to_csv(summary_path, index=False, float_format="%.6f")
    all_rows.to_csv(all_rows_path, index=False, float_format="%.6f")
    changed.to_csv(changed_path, index=False, float_format="%.6f")
    write_text_report(report_path, summary, all_rows, overall, missing)

    print("\nPER-VIDEO METRICS")
    if not summary.empty:
        cols = ["Video", "Windows_Evaluated", "Windows_Changed", "Baseline_MAE_BPM",
                "Experimental_MAE_BPM", "MAE_Improvement_BPM", "Baseline_RMSE_BPM",
                "Experimental_RMSE_BPM", "RMSE_Improvement_BPM", "Baseline_Within_5_BPM_pct",
                "Experimental_Within_5_BPM_pct"]
        print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nPOOLED RESULTS")
    print(f"Evaluated windows: {overall.get('Windows_Evaluated_Pooled', 0)}")
    for name in ["Pooled_Baseline_MAE_BPM", "Pooled_Experimental_MAE_BPM", "Pooled_MAE_Improvement_BPM",
                 "Pooled_Baseline_RMSE_BPM", "Pooled_Experimental_RMSE_BPM", "Pooled_RMSE_Improvement_BPM",
                 "Pooled_Baseline_Within_5_BPM_pct", "Pooled_Experimental_Within_5_BPM_pct"]:
        value = overall.get(name, np.nan)
        print(f"{name}: {value:.3f}" if isinstance(value, (int, float, np.number)) and np.isfinite(value) else f"{name}: {value}")

    print(f"\nChanged window rows: {len(changed)}")
    if not changed.empty:
        cols = ["Video", "Start_sec", "End_sec", "Dominant_Peak_HR_BPM", "Temporally_Selected_HR_BPM",
                "GT_HR_Mean_BPM", "Selected_Candidate_Rank", "Abs_Error_Improvement_BPM"]
        cols = [c for c in cols if c in changed.columns]
        print(changed[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nSaved files:")
    print(f"  Per-video metrics: {summary_path}")
    print(f"  All windows:       {all_rows_path}")
    print(f"  Changed windows:   {changed_path}")
    print(f"  Text report:       {report_path}")
    if errors:
        print("\nErrors encountered:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
