"""Run the same reference-PPG beat diagnostics on a control UBFC recording.

Run from the project folder that contains `results`:
    python check_reference_ppg_control.py --video 3

The script finds the latest `three_signal_comparison.csv` for that video, reads the
matching ground-truth path from `run_summary.txt`, and saves diagnostic CSV/plots in
that result folder. It does not modify pipeline outputs.
"""
from pathlib import Path
import argparse
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, detrend, find_peaks

COMPARISON_FILE = "three_signal_comparison.csv"
LOW_HZ = 0.7
HIGH_HZ = 4.0
WINDOW_SEC = 15.0
HOP_SEC = 5.0
MIN_HR_BPM = 42.0
MAX_HR_BPM = 150.0


def find_latest_comparison(results_dir: Path, video_number: int) -> Path:
    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot find results folder: {results_dir.resolve()}. "
            "Run this script from your project folder or pass --results-dir."
        )
    folder_pattern = re.compile(
        rf"^(?:vid|video)[_-]?0*{video_number}(?:[_-]|$)", re.IGNORECASE
    )
    candidates = [
        p for p in results_dir.rglob(COMPARISON_FILE)
        if folder_pattern.match(p.parent.name)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No {COMPARISON_FILE} found in a result folder for Video {video_number} "
            f"under {results_dir.resolve()}. Check the folder name and that the pipeline ran."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_ground_truth(summary_path: Path):
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}; the pipeline's run_summary.txt is needed.")
    summary_text = summary_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?m)^Ground truth:\s*(.+?)\s*$", summary_text)
    if not match:
        raise ValueError(f"Could not find a 'Ground truth: ...' line in {summary_path}.")
    source_path = Path(match.group(1).strip().strip('"'))
    if not source_path.is_absolute():
        source_path = (summary_path.parent / source_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Cannot find the original ground-truth file: {source_path}")
    gt = pd.read_csv(source_path, header=None)
    if gt.shape[1] != 4:
        raise ValueError(f"Expected 4 columns in {source_path}; found {gt.shape[1]}.")
    gt.columns = ["Timestamp_ms", "HR", "SpO2", "PPG"]
    for col in gt.columns:
        gt[col] = pd.to_numeric(gt[col], errors="coerce")
    gt = gt.dropna(subset=["Timestamp_ms", "HR", "PPG"]).sort_values("Timestamp_ms").reset_index(drop=True)
    gt["Time_sec"] = gt["Timestamp_ms"] / 1000.0
    gt_t = gt["Time_sec"].to_numpy(dtype=float)
    if len(gt) < 100 or np.any(np.diff(gt_t) <= 0):
        raise ValueError("Ground-truth data is too short or timestamps are not strictly increasing.")
    return gt, source_path


def spectral_hr(segment, fs):
    x = np.asarray(segment, dtype=float)
    if len(x) < 3:
        return np.nan
    x = detrend(x, type="linear")
    x = (x - np.mean(x)) * np.hanning(len(x))
    power = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x), d=1.0 / fs)
    valid = np.flatnonzero((freq >= LOW_HZ) & (freq <= HIGH_HZ))
    if not len(valid):
        return np.nan
    idx = int(valid[np.argmax(power[valid])])
    f_peak = float(freq[idx])
    if 0 < idx < len(power) - 1:
        log_power = np.log(power + 1e-12)
        denom = log_power[idx - 1] - 2.0 * log_power[idx] + log_power[idx + 1]
        if abs(denom) > 1e-12:
            delta = float(np.clip(0.5 * (log_power[idx - 1] - log_power[idx + 1]) / denom, -0.5, 0.5))
            f_peak = (idx + delta) * fs / len(x)
    return float(np.clip(f_peak, LOW_HZ, HIGH_HZ) * 60.0)


def autocorrelation_hr(segment, fs):
    x = np.asarray(segment, dtype=float)
    if len(x) < int(fs * 5):
        return np.nan
    x = detrend(x, type="linear")
    x -= np.mean(x)
    if np.std(x) < 1e-10:
        return np.nan
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    if ac[0] <= 0:
        return np.nan
    ac = ac / ac[0]
    min_lag = max(1, int(np.floor(fs * 60.0 / MAX_HR_BPM)))
    max_lag = min(len(ac) - 2, int(np.ceil(fs * 60.0 / MIN_HR_BPM)))
    if max_lag <= min_lag:
        return np.nan
    local = ac[min_lag:max_lag + 1]
    local_peaks, _ = find_peaks(local)
    if len(local_peaks):
        lag = min_lag + int(local_peaks[np.argmax(local[local_peaks])])
    else:
        lag = min_lag + int(np.argmax(local))
    return float(60.0 * fs / lag)


def safe_zscore(x):
    x = np.asarray(x, dtype=float)
    sd = np.std(x)
    return (x - np.mean(x)) / sd if sd > 1e-12 else x - np.mean(x)


def compare_metrics(df):
    rows = []
    gt = df["GT_HR_Mean_BPM"].to_numpy(float)
    for column, label in [
        ("PPG_FFT_HR_BPM", "PPG FFT"),
        ("PPG_PeakInterval_Median_HR_BPM", "PPG median RR"),
        ("PPG_PeakInterval_MeanInterval_HR_BPM", "PPG mean RR"),
        ("PPG_Autocorrelation_HR_BPM", "PPG autocorrelation"),
    ]:
        values = df[column].to_numpy(float)
        mask = np.isfinite(gt) & np.isfinite(values)
        if np.any(mask):
            err = values[mask] - gt[mask]
            rows.append({
                "Method": label,
                "Windows": int(np.sum(mask)),
                "MAE_BPM": float(np.mean(np.abs(err))),
                "RMSE_BPM": float(np.sqrt(np.mean(err ** 2))),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Run reference-PPG beat diagnostics on a control video.")
    parser.add_argument("--video", type=int, default=3, help="Video number to analyze (default: 3).")
    parser.add_argument("--results-dir", default="results", help="Results folder path (default: results).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    comparison_path = find_latest_comparison(results_dir, args.video)
    result_folder = comparison_path.parent
    gt, source_path = read_ground_truth(result_folder / "run_summary.txt")
    gt_t = gt["Time_sec"].to_numpy(float)
    gt_fs = 1.0 / float(np.median(np.diff(gt_t)))
    dt = 1.0 / gt_fs
    uniform_t = np.arange(gt_t[0], gt_t[-1], dt)
    ppg = np.interp(uniform_t, gt_t, gt["PPG"].to_numpy(float))
    ppg = detrend(ppg, type="linear")
    sos = butter(4, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=gt_fs, output="sos")
    ppg_f = sosfiltfilt(sos, ppg)

    min_distance = max(1, int(round(gt_fs * 60.0 / MAX_HR_BPM)))
    prominence = max(0.15 * float(np.std(ppg_f)), 1e-9)
    peak_idxs, _ = find_peaks(ppg_f, distance=min_distance, prominence=prominence)
    peak_t = uniform_t[peak_idxs]

    output_dir = result_folder / "reference_ppg_control_check"
    output_dir.mkdir(exist_ok=True)
    rows = []
    start = 0.0
    overall_end = float(uniform_t[-1])
    while start + WINDOW_SEC <= overall_end + 1e-9:
        end = start + WINDOW_SEC
        p_mask = (uniform_t >= start) & (uniform_t < end)
        gt_mask = (gt_t >= start) & (gt_t < end)
        segment = ppg_f[p_mask]
        local_peak_t = peak_t[(peak_t >= start) & (peak_t < end)]
        rr = np.diff(local_peak_t)
        rr = rr[(rr >= 60.0 / MAX_HR_BPM) & (rr <= 60.0 / MIN_HR_BPM)]
        if len(rr):
            median_rr_hr = float(60.0 / np.median(rr))
            mean_rr_hr = float(60.0 / np.mean(rr))
            rr_hr_std = float(np.std(60.0 / rr, ddof=1)) if len(rr) > 1 else np.nan
        else:
            median_rr_hr = mean_rr_hr = rr_hr_std = np.nan
        gt_hr = gt.loc[gt_mask, "HR"].to_numpy(float)
        fft_hr = spectral_hr(segment, gt_fs)
        rows.append({
            "Start_sec": start,
            "End_sec": end,
            "GT_HR_Mean_BPM": float(np.mean(gt_hr)) if len(gt_hr) else np.nan,
            "GT_HR_Std_BPM": float(np.std(gt_hr)) if len(gt_hr) else np.nan,
            "PPG_FFT_HR_BPM": fft_hr,
            "PPG_PeakInterval_Median_HR_BPM": median_rr_hr,
            "PPG_PeakInterval_MeanInterval_HR_BPM": mean_rr_hr,
            "PPG_PeakInterval_HR_Std_BPM": rr_hr_std,
            "Detected_Peak_Count": int(len(local_peak_t)),
            "Usable_RR_Intervals": int(len(rr)),
            "PPG_Autocorrelation_HR_BPM": autocorrelation_hr(segment, gt_fs),
            "PPG_FFT_vs_GT_Abs_Diff_BPM": abs(fft_hr - float(np.mean(gt_hr))) if len(gt_hr) else np.nan,
        })

        local_t = uniform_t[p_mask]
        local_peak_mask = (peak_t >= start) & (peak_t < end)
        local_peak_idx = peak_idxs[local_peak_mask]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
        z_segment = safe_zscore(segment)
        ax1.plot(local_t, z_segment, label="Filtered reference PPG (z-score)")
        if len(local_peak_idx):
            local_marker_positions = np.clip(np.searchsorted(local_t, uniform_t[local_peak_idx]), 0, len(local_t) - 1)
            ax1.scatter(uniform_t[local_peak_idx], z_segment[local_marker_positions], marker="x", s=35, label="Detected pulse peaks")
        ax1.set_ylabel("Normalized PPG")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")
        ax1.set_title(f"Video {args.video} reference PPG beat check: {start:.0f}-{end:.0f} s")

        if len(gt_hr):
            ax2.plot(gt_t[gt_mask], gt_hr, label="Supplied HR column", linewidth=1.5)
        if len(rr):
            inst_t = (local_peak_t[:-1] + local_peak_t[1:]) / 2.0
            all_rr = np.diff(local_peak_t)
            rr_ok = (all_rr >= 60.0 / MAX_HR_BPM) & (all_rr <= 60.0 / MIN_HR_BPM)
            ax2.plot(inst_t[rr_ok], 60.0 / all_rr[rr_ok], marker="o", markersize=3, label="PPG peak-to-peak HR")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("HR (BPM)")
        ax2.set_ylim(35, 160)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")
        fig.savefig(output_dir / f"video{args.video}_ppg_beats_{int(start):02d}_{int(end):02d}s.png", dpi=140)
        plt.close(fig)
        start += HOP_SEC

    out_df = pd.DataFrame(rows)
    csv_path = output_dir / f"video{args.video}_ppg_beat_check.csv"
    out_df.to_csv(csv_path, index=False, float_format="%.3f")
    metrics_df = compare_metrics(out_df)
    metrics_path = output_dir / f"video{args.video}_ppg_beat_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, float_format="%.3f")

    print(f"VIDEO {args.video} REFERENCE PPG CONTROL CHECK")
    print(f"Comparison file: {comparison_path.resolve()}")
    print(f"Ground-truth source: {source_path.resolve()}")
    print(f"PPG sampling rate estimated from timestamps: {gt_fs:.5f} Hz")
    print(f"Global filtered PPG: {LOW_HZ:.1f}-{HIGH_HZ:.1f} Hz")
    print(f"Global detected pulse peaks: {len(peak_t)}")
    print(f"Peak minimum spacing: {min_distance / gt_fs:.3f} s; prominence threshold: {prominence:.5g}")
    print(f"Saved window CSV: {csv_path.resolve()}")
    print(f"Saved method metrics: {metrics_path.resolve()}")
    print(f"Saved plots: {output_dir.resolve()}")
    print("\nMethod comparison across 15-second windows (5-second hop):")
    print(metrics_df.round(3).to_string(index=False))
    print("\nWindow results:")
    cols = ["Start_sec", "End_sec", "GT_HR_Mean_BPM", "PPG_FFT_HR_BPM", "PPG_PeakInterval_Median_HR_BPM", "PPG_PeakInterval_MeanInterval_HR_BPM", "PPG_Autocorrelation_HR_BPM", "Detected_Peak_Count"]
    print(out_df[cols].round(2).to_string(index=False))
    print("\nInterpret this as a control test, not proof of sensor accuracy. Compare whether methods agree in the good-performing control recording and in Video 4.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
