"""Inspect competing FFT peaks in the existing filtered camera rPPG signal.

Run from the project folder containing `results`:
    python inspect_camera_spectral_candidates.py --video 4

This is diagnostic only. It does not change any pipeline files, filtering,
ROI, POS settings, or the existing HR estimates.
"""
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import detrend, find_peaks

RESULTS_DIR = Path("results")
COMPARISON_FILE = "three_signal_comparison.csv"
FILTERED_FILE = "filtered_rppg.csv"
WINDOW_SEC = 15.0
HOP_SEC = 5.0
LOW_HZ = 0.7
HIGH_HZ = 4.0
MIN_PEAK_SEPARATION_HZ = 0.10
MAX_CANDIDATES = 6


def newest_comparison(video_num: int) -> Path:
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(
            f"Cannot find {RESULTS_DIR.resolve()}. Run this script from the project folder containing 'results'."
        )
    pattern = re.compile(rf"(?:vid|video)[_-]?{video_num}(?:[_-]|$)", re.IGNORECASE)
    paths = [p for p in RESULTS_DIR.rglob(COMPARISON_FILE) if pattern.search(str(p))]
    if not paths:
        raise FileNotFoundError(f"No {COMPARISON_FILE} for Video {video_num} found under {RESULTS_DIR.resolve()}.")
    return max(paths, key=lambda p: p.stat().st_mtime)


def read_fps(summary_path: Path) -> float:
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    txt = summary_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(?m)^FPS:\s*([0-9]+(?:\.[0-9]+)?)", txt)
    if not m:
        raise ValueError(f"Could not find a line like 'FPS: 28.67' in {summary_path}.")
    return float(m.group(1))


def read_gt_path(summary_path: Path) -> Path | None:
    txt = summary_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(?m)^Ground truth:\s*(.+?)\s*$", txt)
    if not m:
        return None
    p = Path(m.group(1).strip().strip('"'))
    if not p.is_absolute():
        p = (summary_path.parent / p).resolve()
    return p if p.is_file() else None


def load_gt(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    # Dataset XMP text is a four-column numeric table: timestamp_ms, HR, SpO2, PPG.
    gt = pd.read_csv(path, header=None)
    if gt.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns in ground-truth file {path}; found {gt.shape[1]}.")
    gt = gt.iloc[:, :4].copy()
    gt.columns = ["Timestamp_ms", "HR", "SpO2", "PPG"]
    for c in gt.columns:
        gt[c] = pd.to_numeric(gt[c], errors="coerce")
    gt = gt.dropna(subset=["Timestamp_ms", "HR"]).sort_values("Timestamp_ms")
    if gt.empty:
        raise ValueError(f"No numeric timestamp/HR rows found in {path}.")
    gt["Time_sec"] = gt["Timestamp_ms"] / 1000.0
    return gt.reset_index(drop=True)


def interp_peak_hr(power: np.ndarray, freq: np.ndarray, idx: int) -> float:
    """Parabolic interpolation around a spectral peak, applied to log-power."""
    if idx <= 0 or idx >= len(power) - 1:
        f = float(freq[idx])
    else:
        lp = np.log(power + 1e-15)
        left, center, right = lp[idx - 1], lp[idx], lp[idx + 1]
        denom = left - 2.0 * center + right
        delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (left - right) / denom
        delta = float(np.clip(delta, -0.5, 0.5))
        df = float(freq[1] - freq[0]) if len(freq) > 1 else 0.0
        f = float(freq[idx] + delta * df)
    return 60.0 * f


def gt_mean_in_window(gt: pd.DataFrame | None, start: float, end: float) -> float:
    if gt is None:
        return np.nan
    mask = (gt["Time_sec"].to_numpy(float) >= start) & (gt["Time_sec"].to_numpy(float) < end)
    vals = gt.loc[mask, "HR"].to_numpy(float)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if len(vals) else np.nan


def candidates_for_segment(segment: np.ndarray, fs: float):
    x = np.asarray(segment, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < int(fs * WINDOW_SEC * 0.95):
        return None, None, None
    x = detrend(x, type="linear")
    x = (x - np.mean(x)) * np.hanning(len(x))
    power = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x), d=1.0 / fs)
    inband = np.flatnonzero((freq >= LOW_HZ) & (freq <= HIGH_HZ))
    if not len(inband):
        return None, None, None
    band_power = power[inband]
    df = float(freq[1] - freq[0]) if len(freq) > 1 else 1.0
    distance_bins = max(1, int(round(MIN_PEAK_SEPARATION_HZ / df)))
    peaks_local, _ = find_peaks(band_power, distance=distance_bins)
    peaks = inband[peaks_local]
    # Always include the highest in-band bin, including cases where it is an endpoint.
    global_peak = int(inband[int(np.argmax(band_power))])
    if global_peak not in peaks:
        peaks = np.append(peaks, global_peak)
    peaks = np.unique(peaks)
    peaks = sorted(peaks.tolist(), key=lambda k: power[k], reverse=True)[:MAX_CANDIDATES]
    rows = []
    max_power = float(power[peaks[0]]) if peaks else np.nan
    for rank, k in enumerate(peaks, start=1):
        rows.append({
            "Peak_Rank": rank,
            "Candidate_HR_BPM": interp_peak_hr(power, freq, int(k)),
            "FFT_Bin_HR_BPM": float(freq[int(k)] * 60.0),
            "Relative_Power_to_Strongest": float(power[int(k)] / max_power) if max_power > 0 else np.nan,
            "Power": float(power[int(k)]),
        })
    return pd.DataFrame(rows), freq, power


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=int, choices=[1, 2, 3, 4, 5, 6, 7], default=4,
                        help="Recording number to inspect (default: 4)")
    args = parser.parse_args()

    comp_path = newest_comparison(args.video)
    folder = comp_path.parent
    summary_path = folder / "run_summary.txt"
    filtered_path = folder / FILTERED_FILE
    if not filtered_path.is_file():
        raise FileNotFoundError(f"Required pipeline output not found: {filtered_path}")
    fps = read_fps(summary_path)
    gt_path = read_gt_path(summary_path)
    gt = load_gt(gt_path)

    camera = pd.read_csv(filtered_path)
    if "Filtered_rPPG" not in camera.columns:
        raise ValueError(f"Expected column 'Filtered_rPPG' in {filtered_path}; found columns: {list(camera.columns)}")
    sig = pd.to_numeric(camera["Filtered_rPPG"], errors="coerce").to_numpy(float)
    if "Frame" in camera.columns:
        frame = pd.to_numeric(camera["Frame"], errors="coerce").to_numpy(float)
        times = (frame - frame[0]) / fps
    else:
        times = np.arange(len(sig), dtype=float) / fps
    valid = np.isfinite(sig) & np.isfinite(times)
    sig, times = sig[valid], times[valid]
    if len(sig) < int(WINDOW_SEC * fps):
        raise ValueError("Filtered camera signal is shorter than one analysis window.")

    out_dir = folder / "camera_spectral_candidates"
    out_dir.mkdir(exist_ok=True)
    max_time = min(float(times[-1]), float(gt["Time_sec"].iloc[-1]) if gt is not None else float(times[-1]))
    starts = np.arange(0.0, max_time - WINDOW_SEC + 1e-9, HOP_SEC)
    all_rows = []
    wide_rows = []

    for start in starts:
        end = start + WINDOW_SEC
        mask = (times >= start) & (times < end)
        segment = sig[mask]
        gt_hr = gt_mean_in_window(gt, start, end)
        candidate_df, freq, power = candidates_for_segment(segment, fps)
        if candidate_df is None:
            continue
        candidates = candidate_df.to_dict("records")
        row = {"Start_sec": float(start), "End_sec": float(end), "GT_HR_Mean_BPM": gt_hr}
        for c in candidates:
            all_rows.append({
                "Start_sec": float(start), "End_sec": float(end), "GT_HR_Mean_BPM": gt_hr,
                **c,
                "Abs_Diff_from_GT_BPM": abs(c["Candidate_HR_BPM"] - gt_hr) if np.isfinite(gt_hr) else np.nan,
            })
            r = int(c["Peak_Rank"])
            row[f"Candidate_{r}_HR_BPM"] = c["Candidate_HR_BPM"]
            row[f"Candidate_{r}_Power_Ratio"] = c["Relative_Power_to_Strongest"]
            row[f"Candidate_{r}_Abs_Diff_GT_BPM"] = abs(c["Candidate_HR_BPM"] - gt_hr) if np.isfinite(gt_hr) else np.nan

        # Existing dominant FFT peak is Candidate 1; show spectral candidate structure, not a new estimator.
        wide_rows.append(row)

        # Keep the whole analysis band on plot, with focus-friendly ticks.
        band = (freq >= LOW_HZ) & (freq <= HIGH_HZ)
        hr_axis = freq[band] * 60.0
        spec = power[band]
        spec_rel = spec / np.max(spec) if len(spec) and np.max(spec) > 0 else spec
        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        ax.plot(hr_axis, spec_rel, linewidth=1.5, label="Camera rPPG relative FFT power")
        if np.isfinite(gt_hr):
            ax.axvline(gt_hr, linestyle="--", linewidth=1.4, label=f"Supplied GT mean: {gt_hr:.1f} BPM")
        for c in candidates[:5]:
            y = np.interp(c["FFT_Bin_HR_BPM"], hr_axis, spec_rel) if len(hr_axis) else 0.0
            ax.scatter([c["Candidate_HR_BPM"]], [y], s=32)
            ax.annotate(f"#{c['Peak_Rank']} {c['Candidate_HR_BPM']:.1f}\nP={c['Relative_Power_to_Strongest']:.2f}",
                        (c["Candidate_HR_BPM"], y), xytext=(3, 7), textcoords="offset points", fontsize=8)
        ax.set_xlim(42, 180)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Frequency expressed as heart rate (BPM)")
        ax.set_ylabel("Power / strongest in-band power")
        ax.set_title(f"Video {args.video}: competing camera FFT peaks, {start:.0f}-{end:.0f} s")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        path = out_dir / f"video{args.video}_spectrum_{int(start):02d}_{int(end):02d}s.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)

    long_df = pd.DataFrame(all_rows)
    wide_df = pd.DataFrame(wide_rows)
    long_path = out_dir / f"video{args.video}_spectral_candidates_long.csv"
    wide_path = out_dir / f"video{args.video}_spectral_candidates_summary.csv"
    long_df.to_csv(long_path, index=False, float_format="%.4f")
    wide_df.to_csv(wide_path, index=False, float_format="%.4f")

    print(f"VIDEO {args.video} CAMERA SPECTRAL CANDIDATE CHECK")
    print(f"Comparison file: {comp_path.resolve()}")
    print(f"Filtered camera signal: {filtered_path.resolve()}")
    print(f"FPS: {fps:.6f}")
    print(f"Spectrum band: {LOW_HZ:.1f}-{HIGH_HZ:.1f} Hz ({LOW_HZ*60:.0f}-{HIGH_HZ*60:.0f} BPM)")
    print(f"Analysis windows: {len(wide_df)}; window={WINDOW_SEC:.0f}s, hop={HOP_SEC:.0f}s")
    print(f"Long candidate CSV: {long_path.resolve()}")
    print(f"Summary CSV: {wide_path.resolve()}")
    print(f"Plots folder: {out_dir.resolve()}")
    cols = ["Start_sec", "End_sec", "GT_HR_Mean_BPM", "Candidate_1_HR_BPM", "Candidate_1_Power_Ratio",
            "Candidate_2_HR_BPM", "Candidate_2_Power_Ratio", "Candidate_3_HR_BPM", "Candidate_3_Power_Ratio"]
    print("\nDominant and competing candidate peaks:")
    available = [c for c in cols if c in wide_df.columns]
    print(wide_df[available].round(3).to_string(index=False))
    print("\nInterpretation: Candidate 1 is the strongest spectral peak, not automatically the true pulse rate.")
    print("Candidate 2+ show competing peaks. GT is displayed only as a comparison line; it is not used to select candidates.")


if __name__ == "__main__":
    main()
