from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
COMPARISON_FILE = "three_signal_comparison.csv"

TIMESTAMPED_FOLDER = re.compile(r"^(?P<video>.+)_(?P<date>\d{8})_(?P<time>\d{6})$")


def metrics_from_errors(errors: np.ndarray) -> dict[str, float | int]:
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]

    if len(errors) == 0:
        return {
            "windows": 0,
            "MAE_BPM": np.nan,
            "RMSE_BPM": np.nan,
            "within_5_count": 0,
            "within_5_percent": np.nan,
        }

    absolute = np.abs(errors)
    within_5 = int(np.sum(absolute <= 5.0))

    return {
        "windows": int(len(errors)),
        "MAE_BPM": float(np.mean(absolute)),
        "RMSE_BPM": float(np.sqrt(np.mean(errors ** 2))),
        "within_5_count": within_5,
        "within_5_percent": float(100.0 * within_5 / len(errors)),
    }


def resolve_video_key(folder: Path) -> tuple[str, str]:
    """Return (video key, sortable run stamp) from timestamped folder name."""
    match = TIMESTAMPED_FOLDER.match(folder.name)
    if match:
        return match.group("video"), match.group("date") + match.group("time")

    # Older/non-timestamped folders are included as a run under their own name.
    return folder.name, str(int(folder.stat().st_mtime))


def main() -> None:
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(
            f"Results directory not found: {RESULTS_DIR}\n"
            "Run main_pipeline_interactive.py first."
        )

    # Find comparison files in run folders.
    candidates: list[tuple[str, str, Path]] = []
    for folder in RESULTS_DIR.iterdir():
        if not folder.is_dir():
            continue
        comparison_path = folder / COMPARISON_FILE
        if not comparison_path.is_file():
            continue
        video_key, run_stamp = resolve_video_key(folder)
        candidates.append((video_key, run_stamp, comparison_path))

    if not candidates:
        raise FileNotFoundError(
            f"No {COMPARISON_FILE} files found under {RESULTS_DIR}.\n"
            "Run the pipeline for each video first, ensuring each run saves its "
            "comparison CSV in its timestamped results folder."
        )

    # Keep only the newest run for each video, avoiding double-counting reruns.
    newest: dict[str, tuple[str, Path]] = {}
    for video_key, run_stamp, csv_path in candidates:
        if video_key not in newest or run_stamp > newest[video_key][0]:
            newest[video_key] = (run_stamp, csv_path)

    per_video_rows: list[dict] = []
    pooled_window_rows: list[pd.DataFrame] = []

    required = {"Camera_rPPG_HR", "Supplied_HR_Mean"}

    for video_key, (run_stamp, csv_path) in sorted(newest.items()):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Skipping unreadable file {csv_path}: {exc}")
            continue

        missing = required - set(df.columns)
        if missing:
            print(
                f"Skipping {csv_path}: missing columns {sorted(missing)}. "
                "Expected the comparison CSV from step9_compare_3_signal.py."
            )
            continue

        camera = pd.to_numeric(df["Camera_rPPG_HR"], errors="coerce").to_numpy()
        gt_mean = pd.to_numeric(df["Supplied_HR_Mean"], errors="coerce").to_numpy()
        valid = np.isfinite(camera) & np.isfinite(gt_mean)
        errors = camera[valid] - gt_mean[valid]

        if not len(errors):
            print(f"Skipping {csv_path}: no valid paired HR values.")
            continue

        row = {
            "Video": video_key,
            "Run_folder": csv_path.parent.name,
            "Comparison_file": str(csv_path),
            **metrics_from_errors(errors),
        }

        # Optional agreement metrics versus reference PPG spectral estimate.
        if "Reference_PPG_HR" in df.columns:
            ppg = pd.to_numeric(df["Reference_PPG_HR"], errors="coerce").to_numpy()
            ppg_valid = np.isfinite(camera) & np.isfinite(ppg)
            ppg_errors = camera[ppg_valid] - ppg[ppg_valid]
            ppg_metrics = metrics_from_errors(ppg_errors)
            row["Camera_vs_PPG_MAE_BPM"] = ppg_metrics["MAE_BPM"]
            row["Camera_vs_PPG_RMSE_BPM"] = ppg_metrics["RMSE_BPM"]

        if "Reference_PPG_HR" in df.columns:
            ppg = pd.to_numeric(df["Reference_PPG_HR"], errors="coerce").to_numpy()
            gt_valid = np.isfinite(ppg) & np.isfinite(gt_mean)
            gt_ppg_errors = ppg[gt_valid] - gt_mean[gt_valid]
            gt_ppg_metrics = metrics_from_errors(gt_ppg_errors)
            row["Reference_PPG_vs_HR_MAE_BPM"] = gt_ppg_metrics["MAE_BPM"]
            row["Reference_PPG_vs_HR_RMSE_BPM"] = gt_ppg_metrics["RMSE_BPM"]

        per_video_rows.append(row)

        # Preserve one window-level table per selected recording for a pooled view.
        windows = df.loc[valid].copy()
        windows.insert(0, "Video", video_key)
        windows["Signed_Error_BPM"] = camera[valid] - gt_mean[valid]
        windows["Absolute_Error_BPM"] = np.abs(windows["Signed_Error_BPM"].to_numpy())
        windows["Within_5_BPM"] = windows["Absolute_Error_BPM"] <= 5.0
        pooled_window_rows.append(windows)

    if not per_video_rows:
        raise RuntimeError("No usable per-video comparison tables were found.")

    per_video_df = pd.DataFrame(per_video_rows).sort_values("Video")
    pooled_df = pd.concat(pooled_window_rows, ignore_index=True)

    per_video_path = RESULTS_DIR / "aggregate_per_video_metrics.csv"
    pooled_path = RESULTS_DIR / "aggregate_window_metrics.csv"

    per_video_df.to_csv(per_video_path, index=False)
    pooled_df.to_csv(pooled_path, index=False)

    pooled_metrics = metrics_from_errors(
        pooled_df["Signed_Error_BPM"].to_numpy(dtype=np.float64)
    )
    macro_mae = float(per_video_df["MAE_BPM"].mean())
    macro_rmse = float(per_video_df["RMSE_BPM"].mean())

    print("\n========== PER-VIDEO METRICS ==========")
    display_cols = [
        col for col in [
            "Video", "windows", "MAE_BPM", "RMSE_BPM",
            "within_5_count", "within_5_percent",
            "Camera_vs_PPG_MAE_BPM", "Camera_vs_PPG_RMSE_BPM"
        ] if col in per_video_df.columns
    ]
    print(per_video_df[display_cols].round(3).to_string(index=False))

    print("\n========== AGGREGATE METRICS ==========")
    print("Videos included:", len(per_video_df))
    print("Total evaluated windows:", pooled_metrics["windows"])
    print("Pooled window MAE:", round(pooled_metrics["MAE_BPM"], 3), "BPM")
    print("Pooled window RMSE:", round(pooled_metrics["RMSE_BPM"], 3), "BPM")
    print(
        "Pooled windows within +/-5 BPM:",
        f"{pooled_metrics['within_5_count']}/{pooled_metrics['windows']}",
        f"({pooled_metrics['within_5_percent']:.1f}%)"
    )
    print("Macro-average per-video MAE:", round(macro_mae, 3), "BPM")
    print("Macro-average per-video RMSE:", round(macro_rmse, 3), "BPM")
    print("=======================================")

    print("\nSaved per-video summary:", per_video_path.resolve())
    print("Saved window-level table:", pooled_path.resolve())

    print(
        "\nNote: HR windows overlap, so pooled windows are correlated. "
        "Treat these metrics as descriptive, not as independent-sample inference."
    )


if __name__ == "__main__":
    main()
