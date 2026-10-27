"""Explore time alignment between Video 4's supplied HR column and reference-PPG beat intervals.

Run from the RPPG project folder (the folder containing `results` and gtdump_4.xmp):
    python check_video4_hr_alignment.py

Read-only: creates CSVs and plots under the latest Video 4 result folder.
The lag scan is diagnostic only; it does not edit or re-run the rPPG pipeline.
"""
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, detrend, find_peaks

RESULTS_DIR = Path("results")
COMPARISON_NAME = "three_signal_comparison.csv"
LOW_HZ = 0.7
HIGH_HZ = 4.0
MIN_HR_BPM = 42.0
MAX_HR_BPM = 150.0
GRID_STEP_SEC = 0.5
SMOOTH_WINDOW_SEC = 5.0
MAX_ABS_LAG_SEC = 10.0
LAG_STEP_SEC = 0.25


def newest_video4_comparison() -> Path:
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(
            f"Cannot find {RESULTS_DIR.resolve()}. Run this script from the project folder containing 'results'."
        )
    candidates = [
        p for p in RESULTS_DIR.rglob(COMPARISON_NAME)
        if re.search(r"(?:vid|video)[_-]?4(?:[_-]|$)", str(p).lower())
    ]
    if not candidates:
        raise FileNotFoundError(f"No Video 4 '{COMPARISON_NAME}' found under {RESULTS_DIR.resolve()}.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_ground_truth(summary_path: Path):
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}. The pipeline run_summary.txt is needed.")
    summary = summary_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?m)^Ground truth:\s*(.+?)\s*$", summary)
    if not match:
        raise ValueError(f"Could not find 'Ground truth: ...' in {summary_path}.")
    source = Path(match.group(1).strip().strip('"'))
    if not source.is_absolute():
        source = (summary_path.parent / source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Cannot find original ground-truth file: {source}")
    gt = pd.read_csv(source, header=None)
    if gt.shape[1] != 4:
        raise ValueError(f"Expected 4 columns in {source} (timestamp ms, HR, SpO2, PPG), found {gt.shape[1]}.")
    gt.columns = ["Timestamp_ms", "HR", "SpO2", "PPG"]
    for col in gt.columns:
        gt[col] = pd.to_numeric(gt[col], errors="coerce")
    gt = gt.dropna(subset=["Timestamp_ms", "HR", "PPG"]).sort_values("Timestamp_ms").reset_index(drop=True)
    gt["Time_sec"] = gt["Timestamp_ms"] / 1000.0
    if len(gt) < 100 or np.any(np.diff(gt["Time_sec"].to_numpy(float)) <= 0):
        raise ValueError("Ground-truth timestamps are too short or not strictly increasing.")
    return gt, source


def metrics(reference, estimate):
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    mask = np.isfinite(ref) & np.isfinite(est)
    if np.count_nonzero(mask) < 3:
        return np.nan, np.nan, np.nan, int(np.count_nonzero(mask))
    a, b = ref[mask], est[mask]
    mae = float(np.mean(np.abs(b - a)))
    rmse = float(np.sqrt(np.mean((b - a) ** 2)))
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else np.nan
    return mae, rmse, corr, int(len(a))


def main():
    comp_path = newest_video4_comparison()
    result_folder = comp_path.parent
    gt, source_path = load_ground_truth(result_folder / "run_summary.txt")

    gt_t = gt["Time_sec"].to_numpy(float)
    median_dt = float(np.median(np.diff(gt_t)))
    fs = 1.0 / median_dt
    uniform_t = np.arange(gt_t[0], gt_t[-1], 1.0 / fs)
    ppg = np.interp(uniform_t, gt_t, gt["PPG"].to_numpy(float))
    ppg = detrend(ppg, type="linear")
    sos = butter(4, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=fs, output="sos")
    ppg_f = sosfiltfilt(sos, ppg)

    min_distance = max(1, int(round(fs * 60.0 / MAX_HR_BPM)))
    prominence = max(0.15 * float(np.std(ppg_f)), 1e-9)
    peak_idxs, _ = find_peaks(ppg_f, distance=min_distance, prominence=prominence)
    peak_t = uniform_t[peak_idxs]
    rr = np.diff(peak_t)
    interval_hr = 60.0 / rr
    midpoint_t = (peak_t[:-1] + peak_t[1:]) / 2.0
    valid = (rr >= 60.0 / MAX_HR_BPM) & (rr <= 60.0 / MIN_HR_BPM)
    midpoint_t = midpoint_t[valid]
    interval_hr = interval_hr[valid]
    if len(midpoint_t) < 10:
        raise ValueError("Too few valid PPG peak-to-peak intervals for the alignment check.")

    # Compare at a regular 0.5 s grid and apply a centered 5 s median to reduce beat-to-beat jitter.
    grid_start = max(float(gt_t[0]), float(midpoint_t[0]))
    grid_end = min(float(gt_t[-1]), float(midpoint_t[-1]))
    grid = np.arange(grid_start, grid_end, GRID_STEP_SEC)
    ppg_hr_grid = np.interp(grid, midpoint_t, interval_hr)
    window_n = max(3, int(round(SMOOTH_WINDOW_SEC / GRID_STEP_SEC)))
    if window_n % 2 == 0:
        window_n += 1
    ppg_hr_smooth = pd.Series(ppg_hr_grid).rolling(window=window_n, center=True, min_periods=max(3, window_n // 2)).median().to_numpy()
    gt_hr_grid = np.interp(grid, gt_t, gt["HR"].to_numpy(float))

    # Use a common interior region for every lag so scores are comparable.
    interior = (grid >= grid_start + MAX_ABS_LAG_SEC) & (grid <= grid_end - MAX_ABS_LAG_SEC)
    base_t = grid[interior]
    base_gt = gt_hr_grid[interior]
    if len(base_t) < 10:
        raise ValueError("Not enough overlapping samples after trimming the lag-search boundaries.")

    lags = np.arange(-MAX_ABS_LAG_SEC, MAX_ABS_LAG_SEC + 1e-9, LAG_STEP_SEC)
    rows = []
    for lag in lags:
        # Positive offset means the PPG-derived HR is sampled later than the GT timestamp.
        aligned_ppg = np.interp(base_t + lag, grid, ppg_hr_smooth, left=np.nan, right=np.nan)
        mae, rmse, corr, count = metrics(base_gt, aligned_ppg)
        rows.append({
            "PPG_Time_Offset_sec": float(lag),
            "MAE_BPM": mae,
            "RMSE_BPM": rmse,
            "Correlation": corr,
            "Compared_Samples": count,
        })
    scan = pd.DataFrame(rows)
    valid_scan = scan.dropna(subset=["MAE_BPM"])
    best_mae_row = valid_scan.loc[valid_scan["MAE_BPM"].idxmin()]
    valid_corr = valid_scan.dropna(subset=["Correlation"])
    best_corr_row = valid_corr.loc[valid_corr["Correlation"].idxmax()] if len(valid_corr) else None

    # Baseline at zero offset, plus the best-MAE candidate.
    zero_idx = int(np.argmin(np.abs(scan["PPG_Time_Offset_sec"].to_numpy(float))))
    zero_row = scan.iloc[zero_idx]
    best_lag = float(best_mae_row["PPG_Time_Offset_sec"])
    best_ppg = np.interp(base_t + best_lag, grid, ppg_hr_smooth, left=np.nan, right=np.nan)

    output_dir = result_folder / "video4_hr_alignment"
    output_dir.mkdir(exist_ok=True)
    samples = pd.DataFrame({
        "Time_sec": grid,
        "Supplied_GT_HR_BPM": gt_hr_grid,
        "PPG_PeakInterval_HR_Raw_BPM": ppg_hr_grid,
        "PPG_PeakInterval_HR_Median5s_BPM": ppg_hr_smooth,
    })
    samples.to_csv(output_dir / "video4_hr_alignment_samples.csv", index=False, float_format="%.3f")
    scan.to_csv(output_dir / "video4_hr_alignment_lag_scan.csv", index=False, float_format="%.4f")

    # Global trace plot.
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.plot(grid, gt_hr_grid, label="Supplied GT HR", linewidth=1.5)
    ax.plot(grid, ppg_hr_grid, label="PPG peak-interval HR (raw)", alpha=0.45, linewidth=0.9)
    ax.plot(grid, ppg_hr_smooth, label="PPG peak-interval HR (5 s median)", linewidth=1.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HR (BPM)")
    ax.set_title("Video 4: supplied HR vs reference-PPG beat-interval HR")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(output_dir / "video4_hr_alignment_global.png", dpi=150)
    plt.close(fig)

    # Compare the series at zero lag and at the best-MAE offset on a shared interior time base.
    zero_ppg = np.interp(base_t, grid, ppg_hr_smooth, left=np.nan, right=np.nan)
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.plot(base_t, base_gt, label="Supplied GT HR", linewidth=1.8)
    ax.plot(base_t, zero_ppg, label="PPG median HR at zero offset", linewidth=1.4, alpha=0.75)
    ax.plot(base_t, best_ppg, label=f"PPG median HR at offset {best_lag:+.2f} s", linewidth=1.6)
    ax.set_xlabel("GT time (s)")
    ax.set_ylabel("HR (BPM)")
    ax.set_title("Video 4: zero-offset and best-MAE alignment (exploratory)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(output_dir / "video4_hr_alignment_best_lag.png", dpi=150)
    plt.close(fig)

    print("VIDEO 4 HR ALIGNMENT CHECK")
    print(f"Comparison file: {comp_path.resolve()}")
    print(f"Ground-truth source: {source_path.resolve()}")
    print(f"Estimated PPG sampling rate: {fs:.5f} Hz")
    print(f"Detected pulse peaks: {len(peak_t)}; usable RR intervals: {len(interval_hr)}")
    print(f"Smoothing: centered {SMOOTH_WINDOW_SEC:.1f} s rolling median")
    print("Lag definition: compare GT HR at time t with PPG-derived HR sampled at t + offset.")
    print(f"Zero offset: MAE={zero_row['MAE_BPM']:.3f} BPM, RMSE={zero_row['RMSE_BPM']:.3f} BPM, correlation={zero_row['Correlation']:.3f}, n={int(zero_row['Compared_Samples'])}")
    print(f"Best-MAE offset: {best_lag:+.2f} s; MAE={best_mae_row['MAE_BPM']:.3f} BPM, RMSE={best_mae_row['RMSE_BPM']:.3f} BPM, correlation={best_mae_row['Correlation']:.3f}, n={int(best_mae_row['Compared_Samples'])}")
    if best_corr_row is not None:
        print(f"Best-correlation offset: {best_corr_row['PPG_Time_Offset_sec']:+.2f} s; correlation={best_corr_row['Correlation']:.3f}, MAE={best_corr_row['MAE_BPM']:.3f} BPM")
    print(f"Saved outputs: {output_dir.resolve()}")
    print("Caution: a best lag found by scanning this one recording is exploratory, not proof of synchronization error.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
