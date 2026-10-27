"""Inspect Video 4 signal spectra for the largest-error 15-second windows.

Run from the project directory containing the `results` folder:
    python inspect_video4_signal_spectra.py

Read-only analysis: it does not modify the original video, ground truth, or pipeline outputs.
It creates a new `video4_signal_diagnostics` folder in the selected result folder.
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
COMPARISON_FILE = "three_signal_comparison.csv"
LOW_HZ = 0.7
HIGH_HZ = 4.0
TOP_WINDOWS_TO_PLOT = 6


def newest_video4_comparison():
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(
            f"Cannot find {RESULTS_DIR.resolve()}. Run this script from the project folder containing 'results'."
        )
    candidates = list(RESULTS_DIR.rglob(COMPARISON_FILE))
    candidates = [
        p for p in candidates
        if re.search(r"(?:vid|video)[_-]?4(?:[_-]|$)", str(p).lower())
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No Video 4 '{COMPARISON_FILE}' found under {RESULTS_DIR.resolve()}."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_summary(summary_path: Path) -> str:
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path.name}. The pipeline summary is needed to obtain FPS and the source ground-truth path."
        )
    return summary_path.read_text(encoding="utf-8", errors="replace")


def read_fps_from_text(content: str, summary_path: Path) -> float:
    match = re.search(r"(?m)^FPS:\s*([0-9]+(?:\.[0-9]+)?)", content)
    if not match:
        raise ValueError(f"Could not read the line 'FPS: ...' from {summary_path}.")
    fps = float(match.group(1))
    if fps <= 0:
        raise ValueError(f"Invalid FPS value in {summary_path}: {fps}")
    return fps


def load_ground_truth(folder: Path, summary_text: str) -> tuple[pd.DataFrame, Path]:
    """Load cleaned GT if available, otherwise read the original .xmp named in run_summary.txt.

    The user's main_pipeline_interactive.py does not create ground_truth_cleaned.csv;
    it passes the original ground-truth file to the comparison stages. This fallback
    is therefore expected for those runs.
    """
    cleaned_path = folder / "ground_truth_cleaned.csv"
    if cleaned_path.is_file():
        gt = pd.read_csv(cleaned_path)
        required = {"Time_sec", "HR", "PPG"}
        if not required.issubset(gt.columns):
            raise ValueError(
                f"{cleaned_path.name} exists but is missing columns: {sorted(required.difference(gt.columns))}"
            )
        return gt, cleaned_path

    match = re.search(r"(?m)^Ground truth:\s*(.+?)\s*$", summary_text)
    if not match:
        raise FileNotFoundError(
            f"{cleaned_path} is not present and no 'Ground truth: ...' path was found in run_summary.txt."
        )
    source_path = Path(match.group(1).strip().strip('"'))
    if not source_path.is_absolute():
        source_path = (folder / source_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"ground_truth_cleaned.csv is not present, and the original ground-truth file from run_summary.txt was not found:\n"
            f"  {source_path}\n"
            "Check that the original gtdump .xmp file still exists at that path."
        )

    gt = pd.read_csv(source_path, header=None)
    if gt.shape[1] != 4:
        raise ValueError(
            f"Expected four columns in the ground-truth file (timestamp ms, HR, SpO2, PPG), "
            f"but found {gt.shape[1]} in {source_path}."
        )
    gt.columns = ["Timestamp_ms", "HR", "SpO2", "PPG"]
    for col in ["Timestamp_ms", "HR", "SpO2", "PPG"]:
        gt[col] = pd.to_numeric(gt[col], errors="coerce")
    gt = gt.dropna(subset=["Timestamp_ms", "HR", "PPG"]).copy()
    gt["Time_sec"] = gt["Timestamp_ms"] / 1000.0
    gt = gt.sort_values("Time_sec").reset_index(drop=True)
    if len(gt) < 3 or np.any(np.diff(gt["Time_sec"].to_numpy(dtype=float)) <= 0):
        raise ValueError(f"Ground-truth timestamps in {source_path} must be strictly increasing.")
    return gt, source_path


def first_existing_column(df: pd.DataFrame, candidates: list[str], description: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not identify the {description} column in three_signal_comparison.csv.\n"
        f"Tried: {candidates}\nPresent columns: {list(df.columns)}"
    )


def fft_spectrum(x, fs):
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        raise ValueError("A window has too few samples for an FFT.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains NaN or infinite values.")
    x = x - np.mean(x)
    xw = x * np.hanning(len(x))
    spectrum = np.fft.rfft(xw)
    freq = np.fft.rfftfreq(len(xw), d=1.0 / fs)
    power = np.abs(spectrum) ** 2
    return freq, power


def dominant_peak(freq, power, fs, n_samples):
    valid = np.flatnonzero((freq >= LOW_HZ) & (freq <= HIGH_HZ))
    if len(valid) == 0:
        return np.nan, np.nan, np.nan
    idx = int(valid[np.argmax(power[valid])])
    peak_freq = float(freq[idx])
    interp_freq = peak_freq
    if 0 < idx < len(power) - 1:
        log_power = np.log(power + 1e-12)
        left, center, right = log_power[idx - 1], log_power[idx], log_power[idx + 1]
        denom = left - 2.0 * center + right
        if abs(denom) > 1e-12:
            delta = 0.5 * (left - right) / denom
            delta = float(np.clip(delta, -0.5, 0.5))
            interp_freq = (idx + delta) * fs / n_samples
    interp_freq = float(np.clip(interp_freq, LOW_HZ, HIGH_HZ))
    return peak_freq * 60.0, interp_freq * 60.0, peak_freq


def top_three_peak_bpm(freq, power):
    band = (freq >= LOW_HZ) & (freq <= HIGH_HZ)
    band_idx = np.flatnonzero(band)
    if len(band_idx) == 0:
        return [np.nan, np.nan, np.nan]
    local_power = power[band]
    local_peaks, _ = find_peaks(local_power)
    if len(local_peaks) == 0:
        local_peaks = np.array([int(np.argmax(local_power))])
    ordered = sorted(local_peaks, key=lambda k: local_power[k], reverse=True)
    bpm = [float(freq[band_idx[k]] * 60.0) for k in ordered[:3]]
    while len(bpm) < 3:
        bpm.append(np.nan)
    return bpm


def zscore(x):
    x = np.asarray(x, dtype=float)
    sd = np.std(x)
    if sd <= 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / sd


def main():
    comp_path = newest_video4_comparison()
    folder = comp_path.parent
    filtered_path = folder / "filtered_rppg.csv"
    summary_path = folder / "run_summary.txt"
    diagnosis_path = folder / "video4_window_diagnosis.csv"
    for needed in (filtered_path, summary_path):
        if not needed.exists():
            raise FileNotFoundError(f"Required pipeline output not found: {needed}")

    summary_text = read_summary(summary_path)
    fps = read_fps_from_text(summary_text, summary_path)
    compare = pd.read_csv(comp_path)
    camera_df = pd.read_csv(filtered_path)
    gt, gt_source_path = load_ground_truth(folder, summary_text)

    # Support the column names used by both the original Step 9 output and the
    # companion video4_window_diagnosis.csv produced in the prior step.
    start_col = first_existing_column(compare, ["Start_sec", "Window_Start_sec", "Window_Start_s", "Start_s"], "window start")
    end_col = first_existing_column(compare, ["End_sec", "Window_End_sec", "Window_End_s", "End_s"], "window end")
    camera_hr_col = first_existing_column(compare, ["Camera_HR_Interpolated", "Camera_rPPG_HR", "Camera_HR_BPM", "HR_Interpolated_BPM"], "camera HR")
    gt_mean_col = first_existing_column(compare, ["GT_Mean_HR", "Supplied_HR_Mean", "GT_Mean_BPM", "GroundTruth_Mean_HR", "Supplied_GT_Mean_HR_BPM"], "supplied ground-truth mean HR")
    # Canonical names make the fallback window ranking independent of Step 9's naming style.
    compare = compare.rename(columns={start_col: "Start_sec", end_col: "End_sec", camera_hr_col: "Camera_HR_Interpolated", gt_mean_col: "GT_Mean_HR"})
    needed_cam = {"Frame", "Filtered_rPPG"}
    missing = needed_cam.difference(camera_df.columns)
    if missing:
        raise ValueError(f"Missing columns in {filtered_path.name}: {sorted(missing)}")
    needed_gt = {"Time_sec", "HR", "PPG"}
    missing = needed_gt.difference(gt.columns)
    if missing:
        raise ValueError(f"Missing columns in ground-truth source {gt_source_path.name}: {sorted(missing)}")

    camera_t = camera_df["Frame"].to_numpy(dtype=float)
    # This pipeline numbers the first frame as 1, so the physical time of frame 1 is 0 s.
    camera_t = (camera_t - camera_t[0]) / fps
    camera_signal = camera_df["Filtered_rPPG"].to_numpy(dtype=float)

    gt = gt.sort_values("Time_sec").reset_index(drop=True)
    gt_t = gt["Time_sec"].to_numpy(dtype=float)
    gt_fs = float(1.0 / np.median(np.diff(gt_t)))
    uniform_t = np.arange(gt_t[0], gt_t[-1], 1.0 / gt_fs)
    uniform_ppg = np.interp(uniform_t, gt_t, gt["PPG"].to_numpy(dtype=float))
    uniform_ppg = detrend(uniform_ppg, type="linear")
    sos = butter(4, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=gt_fs, output="sos")
    filtered_ppg = sosfiltfilt(sos, uniform_ppg)

    # Prefer the diagnosis file's error-ranked rows, if available. Otherwise rank by comparison columns.
    if diagnosis_path.exists():
        ranking = pd.read_csv(diagnosis_path)
        if {"Start_sec", "End_sec"}.issubset(ranking.columns):
            ranking = ranking.sort_values(
                "Camera_vs_GT_Abs_Error_BPM" if "Camera_vs_GT_Abs_Error_BPM" in ranking.columns else "Start_sec",
                ascending=False if "Camera_vs_GT_Abs_Error_BPM" in ranking.columns else True,
            )
            top = ranking.head(TOP_WINDOWS_TO_PLOT)
            selected_windows = top[["Start_sec", "End_sec"]].drop_duplicates().to_records(index=False)
        else:
            selected_windows = []
    else:
        selected_windows = []
    if len(selected_windows) == 0:
        ranked = compare.copy()
        ranked["_abs_err"] = (ranked["Camera_HR_Interpolated"] - ranked["GT_Mean_HR"]).abs()
        ranked = ranked.sort_values("_abs_err", ascending=False).head(TOP_WINDOWS_TO_PLOT)
        selected_windows = ranked[["Start_sec", "End_sec"]].drop_duplicates().to_records(index=False)

    out_dir = folder / "video4_signal_diagnostics"
    out_dir.mkdir(exist_ok=True)
    summary_rows = []

    for start_val, end_val in selected_windows:
        start, end = float(start_val), float(end_val)
        cam_mask = (camera_t >= start) & (camera_t < end)
        ppg_mask = (uniform_t >= start) & (uniform_t < end)
        hr_mask = (gt_t >= start) & (gt_t < end)
        cam_seg = camera_signal[cam_mask]
        ppg_seg = filtered_ppg[ppg_mask]
        gt_window = gt.loc[hr_mask, "HR"].to_numpy(dtype=float)
        if len(cam_seg) < 3 or len(ppg_seg) < 3 or len(gt_window) == 0:
            print(f"Skipping {start:.0f}-{end:.0f}s: insufficient samples.")
            continue

        cam_freq, cam_power = fft_spectrum(cam_seg, fps)
        ppg_freq, ppg_power = fft_spectrum(ppg_seg, gt_fs)
        cam_bin, cam_interp, _ = dominant_peak(cam_freq, cam_power, fps, len(cam_seg))
        ppg_bin, ppg_interp, _ = dominant_peak(ppg_freq, ppg_power, gt_fs, len(ppg_seg))
        cam_top = top_three_peak_bpm(cam_freq, cam_power)
        ppg_top = top_three_peak_bpm(ppg_freq, ppg_power)
        gt_mean = float(np.mean(gt_window))
        gt_median = float(np.median(gt_window))
        gt_std = float(np.std(gt_window))

        summary_rows.append({
            "Start_sec": start,
            "End_sec": end,
            "Center_sec": (start + end) / 2.0,
            "GT_HR_Mean_BPM": gt_mean,
            "GT_HR_Median_BPM": gt_median,
            "GT_HR_Min_BPM": float(np.min(gt_window)),
            "GT_HR_Max_BPM": float(np.max(gt_window)),
            "GT_HR_Std_BPM": gt_std,
            "Camera_FFT_Bin_HR_BPM": cam_bin,
            "Camera_Interpolated_HR_BPM": cam_interp,
            "Camera_Top1_Peak_BPM": cam_top[0],
            "Camera_Top2_Peak_BPM": cam_top[1],
            "Camera_Top3_Peak_BPM": cam_top[2],
            "Reference_PPG_FFT_Bin_HR_BPM": ppg_bin,
            "Reference_PPG_Interpolated_HR_BPM": ppg_interp,
            "Reference_PPG_Top1_Peak_BPM": ppg_top[0],
            "Reference_PPG_Top2_Peak_BPM": ppg_top[1],
            "Reference_PPG_Top3_Peak_BPM": ppg_top[2],
            "Camera_vs_PPG_Abs_Diff_BPM": abs(cam_interp - ppg_interp),
            "Camera_vs_GT_Abs_Diff_BPM": abs(cam_interp - gt_mean),
            "PPG_vs_GT_Abs_Diff_BPM": abs(ppg_interp - gt_mean),
            "Camera_Sample_Count": len(cam_seg),
            "Reference_PPG_Sample_Count": len(ppg_seg),
            "GT_HR_Sample_Count": len(gt_window),
        })

        # Power spectra normalized to their own maximum in the passband, so peak locations are comparable.
        cam_band = (cam_freq >= LOW_HZ) & (cam_freq <= HIGH_HZ)
        ppg_band = (ppg_freq >= LOW_HZ) & (ppg_freq <= HIGH_HZ)
        cam_bpm = cam_freq[cam_band] * 60.0
        ppg_bpm = ppg_freq[ppg_band] * 60.0
        cam_norm_power = cam_power[cam_band] / max(np.max(cam_power[cam_band]), 1e-12)
        ppg_norm_power = ppg_power[ppg_band] / max(np.max(ppg_power[ppg_band]), 1e-12)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
        fig.suptitle(
            f"Video 4 signal diagnosis: {start:.0f}–{end:.0f} s\n"
            f"Camera HR {cam_interp:.2f} BPM | Reference PPG HR {ppg_interp:.2f} BPM | Supplied HR mean {gt_mean:.2f} BPM",
            fontsize=12,
        )

        axes[0].plot(gt_t[hr_mask], gt_window, linewidth=1.1)
        axes[0].axhline(gt_mean, linestyle="--", label=f"Mean = {gt_mean:.2f} BPM")
        axes[0].set_ylabel("Supplied HR (BPM)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_title(f"Supplied HR column in this window (std = {gt_std:.2f} BPM)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(camera_t[cam_mask], zscore(cam_seg), linewidth=0.8, label="Camera filtered rPPG (z-score)")
        axes[1].plot(uniform_t[ppg_mask], zscore(ppg_seg), linewidth=0.8, alpha=0.8, label="Reference filtered PPG (z-score)")
        axes[1].set_ylabel("Normalized amplitude")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_title("Filtered waveforms (normalized separately; phase/morphology need not match)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right", fontsize=8)

        axes[2].plot(cam_bpm, cam_norm_power, marker=".", linewidth=1, label=f"Camera spectrum peak {cam_interp:.1f} BPM")
        axes[2].plot(ppg_bpm, ppg_norm_power, marker=".", linewidth=1, label=f"Reference PPG spectrum peak {ppg_interp:.1f} BPM")
        axes[2].axvline(gt_mean, linestyle="--", label=f"Supplied GT mean {gt_mean:.1f} BPM")
        axes[2].set_xlim(40, 240)
        axes[2].set_ylim(-0.02, 1.05)
        axes[2].set_xlabel("Frequency expressed as heart rate (BPM)")
        axes[2].set_ylabel("Relative power")
        axes[2].set_title("Windowed FFT spectra (each normalized to its own in-band maximum)")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper right", fontsize=8)

        fig_path = out_dir / f"video4_window_{int(start):02d}_{int(end):02d}s.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot: {fig_path}")

    summary_df = pd.DataFrame(summary_rows).sort_values("Start_sec")
    csv_path = out_dir / "video4_signal_spectral_diagnostics.csv"
    summary_df.to_csv(csv_path, index=False, float_format="%.4f")

    print("\nVIDEO 4 SIGNAL/SPECTRUM DIAGNOSTICS COMPLETE")
    print(f"Input result folder: {folder.resolve()}")
    print(f"Video FPS read from run_summary.txt: {fps:.8f}")
    print(f"Ground-truth source: {gt_source_path.resolve()}")
    print(f"Reference PPG sampling rate estimated from timestamps: {gt_fs:.5f} Hz")
    print(f"Plots and CSV saved in: {out_dir.resolve()}")
    if not summary_df.empty:
        cols = [
            "Start_sec", "End_sec", "GT_HR_Mean_BPM", "GT_HR_Std_BPM",
            "Camera_Interpolated_HR_BPM", "Camera_Top2_Peak_BPM",
            "Reference_PPG_Interpolated_HR_BPM", "Reference_PPG_Top2_Peak_BPM",
            "Camera_vs_PPG_Abs_Diff_BPM", "Camera_vs_GT_Abs_Diff_BPM", "PPG_vs_GT_Abs_Diff_BPM",
        ]
        print("\nDiagnostic summary:")
        print(summary_df[cols].round(3).to_string(index=False))
        print("\nInterpretation notes:")
        print("- A strong secondary peak near the dominant peak can make the HR estimate unstable.")
        print("- A large GT_HR_Std_BPM means the supplied HR changes within the 15-second window; its mean may not equal the spectrum's dominant HR.")
        print("- Similar camera/PPG spectral peaks with both far from supplied HR suggests an HR-column or timing/definition discrepancy, not necessarily a bad camera signal.")
        print("- These are diagnostic clues, not proof; the signal waveform phase is not expected to match across sensors.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
