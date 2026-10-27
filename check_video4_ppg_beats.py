"""Independently check Video 4 reference PPG using beat intervals and autocorrelation.

Run from the project folder that contains `results` and the original gtdump_4.xmp:
    python check_video4_ppg_beats.py

This is read-only with respect to pipeline outputs. It creates a subfolder beside the
latest Video 4 comparison CSV containing a CSV summary and plots for 15-second windows.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, detrend, find_peaks

RESULTS_DIR = Path("results")
COMPARISON_FILE = "three_signal_comparison.csv"
LOW_HZ = 0.7
HIGH_HZ = 4.0
WINDOW_SEC = 15.0
HOP_SEC = 5.0
# Ground-truth HR for Video 4 is about 62-74 BPM; allow a wider range during diagnostics.
MIN_HR_BPM = 42.0
MAX_HR_BPM = 150.0


def newest_video4_comparison():
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(
            f"Cannot find {RESULTS_DIR.resolve()}. Run this script from the project folder containing 'results'."
        )
    candidates = list(RESULTS_DIR.rglob(COMPARISON_FILE))
    candidates = [p for p in candidates if re.search(r"(?:vid|video)[_-]?4(?:[_-]|$)", str(p).lower())]
    if not candidates:
        raise FileNotFoundError(f"No Video 4 '{COMPARISON_FILE}' found under {RESULTS_DIR.resolve()}.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_summary(summary_path):
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run the pipeline first.")
    return summary_path.read_text(encoding="utf-8", errors="replace")


def load_gt(summary_path, summary_text):
    match = re.search(r"(?m)^Ground truth:\s*(.+?)\s*$", summary_text)
    if not match:
        raise ValueError(f"Could not find 'Ground truth: ...' in {summary_path}.")
    source_path = Path(match.group(1).strip().strip('"'))
    if not source_path.is_absolute():
        source_path = (summary_path.parent / source_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Cannot find original ground-truth file: {source_path}")
    gt = pd.read_csv(source_path, header=None)
    if gt.shape[1] != 4:
        raise ValueError(f"Expected four GT columns; found {gt.shape[1]} in {source_path}.")
    gt.columns = ["Timestamp_ms", "HR", "SpO2", "PPG"]
    for c in gt.columns:
        gt[c] = pd.to_numeric(gt[c], errors="coerce")
    gt = gt.dropna(subset=["Timestamp_ms", "HR", "PPG"]).sort_values("Timestamp_ms").reset_index(drop=True)
    gt["Time_sec"] = gt["Timestamp_ms"] / 1000.0
    if len(gt) < 100 or np.any(np.diff(gt["Time_sec"].to_numpy()) <= 0):
        raise ValueError("GT data is too short or timestamps are not strictly increasing.")
    return gt, source_path


def spectral_hr(segment, fs):
    segment = np.asarray(segment, dtype=float)
    if len(segment) < 3:
        return np.nan
    segment = detrend(segment, type="linear")
    segment = (segment - np.mean(segment)) * np.hanning(len(segment))
    power = np.abs(np.fft.rfft(segment)) ** 2
    freq = np.fft.rfftfreq(len(segment), d=1.0 / fs)
    valid = np.flatnonzero((freq >= LOW_HZ) & (freq <= HIGH_HZ))
    if len(valid) == 0:
        return np.nan
    idx = int(valid[np.argmax(power[valid])])
    # Log-power parabolic interpolation around the maximum FFT bin.
    f_peak = float(freq[idx])
    if 0 < idx < len(power) - 1:
        lp = np.log(power + 1e-12)
        den = lp[idx - 1] - 2.0 * lp[idx] + lp[idx + 1]
        if abs(den) > 1e-12:
            delta = float(np.clip(0.5 * (lp[idx - 1] - lp[idx + 1]) / den, -0.5, 0.5))
            f_peak = (idx + delta) * fs / len(segment)
    return float(np.clip(f_peak, LOW_HZ, HIGH_HZ) * 60.0)


def autocorrelation_hr(segment, fs):
    x = np.asarray(segment, dtype=float)
    if len(x) < int(fs * 5):
        return np.nan
    x = detrend(x, type="linear")
    x = x - np.mean(x)
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
    loc_peaks, _ = find_peaks(local)
    if len(loc_peaks):
        # Select the strongest local autocorrelation maximum in the plausible HR band.
        lag = min_lag + int(loc_peaks[np.argmax(local[loc_peaks])])
    else:
        lag = min_lag + int(np.argmax(local))
    return 60.0 * fs / lag


def safe_zscore(x):
    x = np.asarray(x, dtype=float)
    sd = np.std(x)
    return (x - np.mean(x)) / sd if sd > 1e-12 else x - np.mean(x)


def main():
    comp_path = newest_video4_comparison()
    folder = comp_path.parent
    summary_path = folder / "run_summary.txt"
    summary_text = read_summary(summary_path)
    gt, source_path = load_gt(summary_path, summary_text)

    gt_t = gt["Time_sec"].to_numpy(dtype=float)
    gt_fs = 1.0 / float(np.median(np.diff(gt_t)))
    dt = 1.0 / gt_fs
    uniform_t = np.arange(gt_t[0], gt_t[-1], dt)
    ppg = np.interp(uniform_t, gt_t, gt["PPG"].to_numpy(dtype=float))
    ppg = detrend(ppg, type="linear")
    sos = butter(4, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=gt_fs, output="sos")
    ppg_f = sosfiltfilt(sos, ppg)

    # Beat peaks are detected on the whole filtered trace so windows don't reset the detector.
    # The prominence threshold is intentionally modest; the output reports beat counts so poor
    # detections can be identified rather than silently trusted.
    min_distance = max(1, int(round(gt_fs * 60.0 / MAX_HR_BPM)))
    prominence = max(0.15 * float(np.std(ppg_f)), 1e-9)
    peaks, properties = find_peaks(ppg_f, distance=min_distance, prominence=prominence)
    peak_t = uniform_t[peaks]

    output_dir = folder / "video4_ppg_beat_check"
    output_dir.mkdir(exist_ok=True)
    rows = []
    start = 0.0
    overall_end = float(uniform_t[-1])
    while start + WINDOW_SEC <= overall_end + 1e-9:
        end = start + WINDOW_SEC
        pmask = (uniform_t >= start) & (uniform_t < end)
        hmask = (gt_t >= start) & (gt_t < end)
        seg = ppg_f[pmask]
        local_peak_times = peak_t[(peak_t >= start) & (peak_t < end)]
        rr = np.diff(local_peak_times)
        rr = rr[(rr >= 60.0 / MAX_HR_BPM) & (rr <= 60.0 / MIN_HR_BPM)]
        if len(rr):
            beat_hr_median = float(60.0 / np.median(rr))
            beat_hr_mean_interval = float(60.0 / np.mean(rr))
            beat_hr_sd = float(np.std(60.0 / rr, ddof=1)) if len(rr) > 1 else np.nan
        else:
            beat_hr_median = beat_hr_mean_interval = beat_hr_sd = np.nan
        gt_hr = gt.loc[hmask, "HR"].to_numpy(dtype=float)
        rows.append({
            "Start_sec": start,
            "End_sec": end,
            "GT_HR_Mean_BPM": float(np.mean(gt_hr)) if len(gt_hr) else np.nan,
            "GT_HR_Std_BPM": float(np.std(gt_hr)) if len(gt_hr) else np.nan,
            "PPG_FFT_HR_BPM": spectral_hr(seg, gt_fs),
            "PPG_PeakInterval_Median_HR_BPM": beat_hr_median,
            "PPG_PeakInterval_MeanInterval_HR_BPM": beat_hr_mean_interval,
            "PPG_PeakInterval_HR_Std_BPM": beat_hr_sd,
            "Detected_Peak_Count": int(len(local_peak_times)),
            "Usable_RR_Intervals": int(len(rr)),
            "PPG_Autocorrelation_HR_BPM": autocorrelation_hr(seg, gt_fs),
            "PPG_vs_GT_FFT_Abs_Diff_BPM": abs(spectral_hr(seg, gt_fs) - float(np.mean(gt_hr))) if len(gt_hr) else np.nan,
        })

        # Plot raw ground-truth HR and filtered PPG with detected beat markers.
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
        local_t = uniform_t[pmask]
        ax1.plot(local_t, safe_zscore(seg), label="Filtered reference PPG (z-score)")
        local_peaks_mask = (peak_t >= start) & (peak_t < end)
        local_idxs = peaks[local_peaks_mask]
        if len(local_idxs):
            ax1.scatter(uniform_t[local_idxs], safe_zscore(ppg_f[pmask])[np.searchsorted(local_t, uniform_t[local_idxs])],
                        marker="x", s=35, label="Detected pulse peaks")
        ax1.set_ylabel("Normalized PPG")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")
        ax1.set_title(f"Video 4 reference PPG beat check: {start:.0f}-{end:.0f} s")

        if len(gt_hr):
            ax2.plot(gt_t[hmask], gt_hr, label="Supplied HR column", linewidth=1.5)
        if len(rr):
            inst_t = (local_peak_times[:-1] + local_peak_times[1:]) / 2.0
            inst_hr = 60.0 / np.diff(local_peak_times)
            rr_mask = (np.diff(local_peak_times) >= 60.0 / MAX_HR_BPM) & (np.diff(local_peak_times) <= 60.0 / MIN_HR_BPM)
            ax2.plot(inst_t[rr_mask], inst_hr[rr_mask], marker="o", markersize=3, label="PPG peak-to-peak HR")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("HR (BPM)")
        ax2.set_ylim(35, 160)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")
        fig.savefig(output_dir / f"video4_ppg_beats_{int(start):02d}_{int(end):02d}s.png", dpi=140)
        plt.close(fig)
        start += HOP_SEC

    out_df = pd.DataFrame(rows)
    csv_path = output_dir / "video4_ppg_beat_check.csv"
    out_df.to_csv(csv_path, index=False, float_format="%.3f")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print("VIDEO 4 REFERENCE PPG BEAT CHECK")
    print(f"Comparison file: {comp_path.resolve()}")
    print(f"Ground-truth source: {source_path.resolve()}")
    print(f"PPG sampling rate estimated from timestamps: {gt_fs:.5f} Hz")
    print(f"Global PPG bandpass: {LOW_HZ:.1f}-{HIGH_HZ:.1f} Hz")
    print(f"Global detected pulse peaks: {len(peaks)}")
    print(f"Peak detector minimum spacing: {min_distance / gt_fs:.3f} s; prominence threshold: {prominence:.5g}")
    print(f"Saved CSV: {csv_path.resolve()}")
    print(f"Saved plots in: {output_dir.resolve()}")
    print("\n15-second windows, 5-second hop:")
    print(out_df.round(2).to_string(index=False))
    print("\nHow to interpret: compare PPG_FFT_HR_BPM with the two peak-interval methods and autocorrelation.")
    print("If the methods agree with each other but differ from GT_HR_Mean_BPM, that supports a disagreement between the supplied HR column and the waveform-derived rate.")
    print("If methods disagree with each other or peak counts look implausible, pulse-peak detection is not reliable enough to draw that conclusion.")


if __name__ == "__main__":
    main()
