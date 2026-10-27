"""Diagnose the filtered camera rPPG waveform with beat intervals and autocorrelation.

Run from the user's RPPG project folder, which contains `results`:
    python check_camera_rppg_beats.py --video 4

For a control recording, the same script can be run as:
    python check_camera_rppg_beats.py --video 3

This is diagnostic only: it does not modify the existing pipeline outputs or settings.
"""
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend

RESULTS_DIR = Path("results")
COMPARISON_FILE = "three_signal_comparison.csv"
FILTERED_FILE = "filtered_rppg.csv"
WINDOW_SEC = 15.0
HOP_SEC = 5.0
LOW_HZ = 0.7
HIGH_HZ = 4.0
MIN_HR_BPM = 42.0
MAX_HR_BPM = 150.0


def find_newest_comparison(video_num: int) -> Path:
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(
            f"Cannot find {RESULTS_DIR.resolve()}. Run from the project folder containing 'results'."
        )
    pat = re.compile(rf"(?:vid|video)[_-]?{video_num}(?:[_-]|$)", re.IGNORECASE)
    candidates = [p for p in RESULTS_DIR.rglob(COMPARISON_FILE) if pat.search(str(p))]
    if not candidates:
        raise FileNotFoundError(
            f"No '{COMPARISON_FILE}' for Video {video_num} found under {RESULTS_DIR.resolve()}."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_summary(summary_path: Path):
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}; run_summary.txt is needed.")
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    m_fps = re.search(r"(?m)^FPS:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not m_fps:
        raise ValueError(f"Could not find 'FPS: ...' in {summary_path}.")
    fps = float(m_fps.group(1))
    m_gt = re.search(r"(?m)^Ground truth:\s*(.+?)\s*$", text)
    if not m_gt:
        raise ValueError(f"Could not find 'Ground truth: ...' in {summary_path}.")
    gt_path = Path(m_gt.group(1).strip().strip('"'))
    if not gt_path.is_absolute():
        gt_path = (summary_path.parent / gt_path).resolve()
    if not gt_path.is_file():
        raise FileNotFoundError(f"Cannot find original ground-truth file: {gt_path}")
    return fps, gt_path, text


def load_gt(gt_path: Path) -> pd.DataFrame:
    gt = pd.read_csv(gt_path, header=None)
    if gt.shape[1] != 4:
        raise ValueError(f"Expected 4 columns in {gt_path}; found {gt.shape[1]}.")
    gt.columns = ["Timestamp_ms", "HR", "SpO2", "PPG"]
    for c in gt.columns:
        gt[c] = pd.to_numeric(gt[c], errors="coerce")
    gt = gt.dropna(subset=["Timestamp_ms", "HR", "PPG"]).copy()
    gt["Time_sec"] = gt["Timestamp_ms"] / 1000.0
    gt = gt.sort_values("Time_sec").reset_index(drop=True)
    if len(gt) < 3 or np.any(np.diff(gt["Time_sec"].to_numpy(dtype=float)) <= 0):
        raise ValueError(f"Ground-truth timestamps in {gt_path} are invalid or not strictly increasing.")
    return gt


def pick_column(df: pd.DataFrame, names, description):
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(
        f"Cannot identify {description}. Tried {names}; available columns: {list(df.columns)}"
    )


def spectral_hr(segment, fs):
    x = np.asarray(segment, dtype=float)
    if len(x) < 3 or not np.all(np.isfinite(x)):
        return np.nan
    x = detrend(x, type="linear")
    x = (x - np.mean(x)) * np.hanning(len(x))
    power = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x), d=1.0 / fs)
    valid = np.flatnonzero((freq >= LOW_HZ) & (freq <= HIGH_HZ))
    if len(valid) == 0:
        return np.nan
    k = int(valid[np.argmax(power[valid])])
    f = float(freq[k])
    # Parabolic interpolation on log-power around the dominant peak.
    if 0 < k < len(power) - 1:
        lp = np.log(power + 1e-12)
        den = lp[k - 1] - 2 * lp[k] + lp[k + 1]
        if abs(den) > 1e-12:
            delta = float(np.clip(0.5 * (lp[k - 1] - lp[k + 1]) / den, -0.5, 0.5))
            f = (k + delta) * fs / len(x)
    return float(np.clip(f, LOW_HZ, HIGH_HZ) * 60.0)


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
    ac /= ac[0]
    min_lag = max(1, int(np.floor(fs * 60.0 / MAX_HR_BPM)))
    max_lag = min(len(ac) - 2, int(np.ceil(fs * 60.0 / MIN_HR_BPM)))
    if max_lag <= min_lag:
        return np.nan
    band = ac[min_lag:max_lag + 1]
    loc, _ = find_peaks(band)
    lag = min_lag + int(loc[np.argmax(band[loc])]) if len(loc) else min_lag + int(np.argmax(band))
    return float(60.0 * fs / lag)


def metric_row(y_true, y_est):
    mask = np.isfinite(y_true) & np.isfinite(y_est)
    if not np.any(mask):
        return {"Windows": 0, "MAE_BPM": np.nan, "RMSE_BPM": np.nan, "Bias_BPM": np.nan}
    err = y_est[mask] - y_true[mask]
    return {
        "Windows": int(mask.sum()),
        "MAE_BPM": float(np.mean(np.abs(err))),
        "RMSE_BPM": float(np.sqrt(np.mean(err ** 2))),
        "Bias_BPM": float(np.mean(err)),
    }


def safe_zscore(x):
    x = np.asarray(x, dtype=float)
    sd = np.std(x)
    return (x - np.mean(x)) / sd if sd > 1e-12 else x - np.mean(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=int, choices=[1, 2, 3, 4, 5], default=4,
                        help="recording number to inspect (default: 4)")
    args = parser.parse_args()

    comp_path = find_newest_comparison(args.video)
    folder = comp_path.parent
    filtered_path = folder / FILTERED_FILE
    summary_path = folder / "run_summary.txt"
    if not filtered_path.exists():
        raise FileNotFoundError(f"Required pipeline output not found: {filtered_path}")
    fps, gt_path, _ = parse_summary(summary_path)
    gt = load_gt(gt_path)
    comp = pd.read_csv(comp_path)

    time_col = pick_column(comp, ["Start_sec", "Window_Start_sec", "Window_Start_s", "Start_s"], "comparison window start")
    camera_hr_col = pick_column(comp, ["Camera_HR_Interpolated", "Camera_rPPG_HR", "Camera_HR_BPM", "HR_Interpolated_BPM"], "existing camera HR estimate")
    comp = comp.rename(columns={time_col: "Start_sec", camera_hr_col: "Existing_Camera_FFT_HR_BPM"})

    camera_df = pd.read_csv(filtered_path)
    if "Filtered_rPPG" not in camera_df.columns:
        raise ValueError(f"Expected 'Filtered_rPPG' in {filtered_path}; columns are {list(camera_df.columns)}")
    if "Frame" in camera_df.columns:
        ct = (camera_df["Frame"].to_numpy(dtype=float) - float(camera_df["Frame"].iloc[0])) / fps
    else:
        ct = np.arange(len(camera_df), dtype=float) / fps
    sig = pd.to_numeric(camera_df["Filtered_rPPG"], errors="coerce").to_numpy(dtype=float)
    valid_samples = np.isfinite(ct) & np.isfinite(sig)
    ct, sig = ct[valid_samples], sig[valid_samples]
    if len(sig) < int(fps * WINDOW_SEC):
        raise ValueError("Filtered camera rPPG signal is shorter than one 15-second window.")

    # The signal is already band-passed by the pipeline. Do not filter it again.
    min_distance = max(1, int(round(fps * 60.0 / MAX_HR_BPM)))
    prominence = max(0.15 * float(np.std(sig)), 1e-9)
    peak_idxs, props = find_peaks(sig, distance=min_distance, prominence=prominence)
    peak_t = ct[peak_idxs]

    out_dir = folder / "camera_rppg_beat_check"
    out_dir.mkdir(exist_ok=True)
    gt_t = gt["Time_sec"].to_numpy(dtype=float)
    gt_hr = gt["HR"].to_numpy(dtype=float)
    rows = []
    starts = np.arange(0.0, min(float(ct[-1]), float(gt_t[-1])) - WINDOW_SEC + 1e-9, HOP_SEC)
    for start in starts:
        end = start + WINDOW_SEC
        smask = (ct >= start) & (ct < end)
        hmask = (gt_t >= start) & (gt_t < end)
        local_peak_t = peak_t[(peak_t >= start) & (peak_t < end)]
        rr = np.diff(local_peak_t)
        rr = rr[(rr >= 60.0 / MAX_HR_BPM) & (rr <= 60.0 / MIN_HR_BPM)]
        med_rr_hr = float(60.0 / np.median(rr)) if len(rr) else np.nan
        mean_rr_hr = float(60.0 / np.mean(rr)) if len(rr) else np.nan
        segment = sig[smask]
        this_gt = float(np.mean(gt_hr[hmask])) if np.any(hmask) else np.nan
        rows.append({
            "Start_sec": float(start), "End_sec": float(end),
            "GT_HR_Mean_BPM": this_gt,
            "Existing_Camera_FFT_HR_BPM": np.nan,
            "Camera_FFT_HR_BPM": spectral_hr(segment, fps),
            "Camera_PeakInterval_Median_HR_BPM": med_rr_hr,
            "Camera_PeakInterval_MeanInterval_HR_BPM": mean_rr_hr,
            "Camera_Autocorrelation_HR_BPM": autocorrelation_hr(segment, fps),
            "Detected_Peak_Count": int(len(local_peak_t)),
            "Usable_RR_Intervals": int(len(rr)),
            "PeakInterval_HR_Std_BPM": float(np.std(60.0 / rr, ddof=1)) if len(rr) > 1 else np.nan,
        })
        # Two-panel plot: detected peaks and interval HR vs supplied GT HR.
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
        local_t = ct[smask]
        local_sig = sig[smask]
        ax1.plot(local_t, safe_zscore(local_sig), label="Filtered camera rPPG (z-score)")
        local_peak_mask = (peak_t >= start) & (peak_t < end)
        local_peak_times = peak_t[local_peak_mask]
        if len(local_peak_times):
            local_peak_values = np.interp(local_peak_times, ct, sig)
            ax1.scatter(local_peak_times, (local_peak_values - np.mean(local_sig)) / (np.std(local_sig) + 1e-12),
                        marker="x", s=35, label="Detected peaks")
        ax1.set_ylabel("Normalized rPPG")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")
        ax1.set_title(f"Video {args.video} camera rPPG beat check: {start:.0f}-{end:.0f} s")

        if np.any(hmask):
            ax2.plot(gt_t[hmask], gt_hr[hmask], label="Supplied HR column", linewidth=1.5)
        if len(local_peak_times) >= 2:
            intervals = np.diff(local_peak_times)
            inst_hr = 60.0 / intervals
            inst_t = (local_peak_times[:-1] + local_peak_times[1:]) / 2.0
            ok = (intervals >= 60.0 / MAX_HR_BPM) & (intervals <= 60.0 / MIN_HR_BPM)
            ax2.plot(inst_t[ok], inst_hr[ok], marker="o", markersize=3, label="Camera peak-interval HR")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("HR (BPM)")
        ax2.set_ylim(35, 160)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")
        fig.savefig(out_dir / f"video{args.video}_camera_beats_{int(start):02d}_{int(end):02d}s.png", dpi=140)
        plt.close(fig)

    result = pd.DataFrame(rows)
    # Add existing Step 9 camera HR by matching start time, if available.
    if "Start_sec" in comp.columns and "Existing_Camera_FFT_HR_BPM" in comp.columns:
        map_est = comp.drop_duplicates("Start_sec").set_index("Start_sec")["Existing_Camera_FFT_HR_BPM"]
        result["Existing_Camera_FFT_HR_BPM"] = result["Start_sec"].map(map_est)

    csv_path = out_dir / f"video{args.video}_camera_beat_check.csv"
    metrics_path = out_dir / f"video{args.video}_camera_beat_metrics.csv"
    result.to_csv(csv_path, index=False, float_format="%.3f")

    method_columns = [
        ("Existing pipeline camera FFT", "Existing_Camera_FFT_HR_BPM"),
        ("Windowed camera FFT", "Camera_FFT_HR_BPM"),
        ("Camera median RR", "Camera_PeakInterval_Median_HR_BPM"),
        ("Camera mean RR", "Camera_PeakInterval_MeanInterval_HR_BPM"),
        ("Camera autocorrelation", "Camera_Autocorrelation_HR_BPM"),
    ]
    metric_rows = []
    truth = result["GT_HR_Mean_BPM"].to_numpy(dtype=float)
    for label, col in method_columns:
        if col in result:
            metric_rows.append({"Method": label, **metric_row(truth, result[col].to_numpy(dtype=float))})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(metrics_path, index=False, float_format="%.3f")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print(f"VIDEO {args.video} CAMERA rPPG BEAT CHECK")
    print(f"Comparison file: {comp_path.resolve()}")
    print(f"Filtered camera signal: {filtered_path.resolve()}")
    print(f"Ground-truth source: {gt_path.resolve()}")
    print(f"Video sampling rate: {fps:.5f} Hz")
    print("Signal used: existing filtered_rPPG.csv (not filtered again)")
    print(f"Global detected peaks: {len(peak_idxs)}")
    print(f"Peak minimum spacing: {min_distance / fps:.3f} s; prominence threshold: {prominence:.5g}")
    print(f"Saved window CSV: {csv_path.resolve()}")
    print(f"Saved method metrics: {metrics_path.resolve()}")
    print(f"Saved plots: {out_dir.resolve()}")
    print("\nMethod comparison across 15-second windows (5-second hop):")
    print(metrics.round(3).to_string(index=False))
    print("\nWindow results:")
    print(result.round(3).to_string(index=False))
    print("\nInterpretation caution: the POS rPPG signal may not have pulse morphology as clean as a contact PPG waveform.")
    print("A beat detector is only a diagnostic; keep your baseline FFT estimate and pipeline parameters unchanged for now.")


if __name__ == "__main__":
    main()
