from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, detrend


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = BASE_DIR / "vid_4.avi"
RPPG_PATH = BASE_DIR / "filtered_rppg.csv"
GT_PATH = BASE_DIR / "gtdump_4.xmp"

LOW_FREQ = 0.7
HIGH_FREQ = 4.0

# Intervals where previous estimates showed disagreement.
PROBLEM_WINDOWS = [
    (0, 15),
    (20, 35),
    (55, 70)
]


# ============================================================
# 2. VERIFY INPUT FILES
# ============================================================

for path in [VIDEO_PATH, RPPG_PATH, GT_PATH]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path.resolve()}"
        )

print("Video:", VIDEO_PATH.resolve())
print("Filtered rPPG:", RPPG_PATH.resolve())
print("Ground truth:", GT_PATH.resolve())


# ============================================================
# 3. GET VIDEO FRAME RATE
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open {VIDEO_PATH}")

FPS = cap.get(cv2.CAP_PROP_FPS)
VIDEO_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()

VIDEO_DURATION = VIDEO_FRAMES / FPS

print("\nVideo FPS:", FPS)
print("Video frames:", VIDEO_FRAMES)
print("Video duration:", VIDEO_DURATION, "seconds")


# ============================================================
# 4. LOAD CAMERA-DERIVED rPPG
# ============================================================

rppg_df = pd.read_csv(RPPG_PATH)

if "Filtered_rPPG" not in rppg_df.columns:
    raise ValueError(
        "filtered_rppg.csv requires a Filtered_rPPG column."
    )

rppg = rppg_df["Filtered_rPPG"].to_numpy(dtype=float)

if len(rppg) != VIDEO_FRAMES:
    raise ValueError(
        "rPPG sample count does not match video frame count. "
        "Check that both belong to Video 4."
    )


# ============================================================
# 5. LOAD REFERENCE PPG AND SUPPLIED HR
# ============================================================

gt = pd.read_csv(GT_PATH, header=None)

if gt.shape[1] != 4:
    raise ValueError(
        f"Expected 4 ground-truth columns; got {gt.shape[1]}"
    )

gt.columns = ["Timestamp_ms", "HR", "SpO2", "PPG"]

gt_time = gt["Timestamp_ms"].to_numpy(dtype=float) / 1000.0
gt_hr = gt["HR"].to_numpy(dtype=float)
gt_ppg = gt["PPG"].to_numpy(dtype=float)

# Estimate reference sampling frequency from timestamps.
GT_FS = 1.0 / np.median(np.diff(gt_time))

print("\nReference PPG sampling rate:", GT_FS, "Hz")
print("Ground-truth HR mean:", gt_hr.mean(), "BPM")


# ============================================================
# 6. FILTER THE FULL REFERENCE PPG
# ============================================================

# Create a uniform time grid.
uniform_time = np.arange(
    gt_time[0],
    gt_time[-1],
    1.0 / GT_FS
)

# Interpolate reference PPG onto that grid.
uniform_ppg = np.interp(
    uniform_time,
    gt_time,
    gt_ppg
)

# Remove slow overall linear trend.
uniform_ppg = detrend(
    uniform_ppg,
    type="linear"
)

# Filter the complete reference recording before extracting
# analysis windows to reduce window-edge filtering effects.
sos = butter(
    4,
    [LOW_FREQ, HIGH_FREQ],
    btype="bandpass",
    fs=GT_FS,
    output="sos"
)

filtered_reference_ppg = sosfiltfilt(
    sos,
    uniform_ppg
)


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def standardize(signal):
    """Center and scale a signal for visual comparison."""
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)

    std = np.std(signal)

    if std > 0:
        signal = signal / std

    return signal


def calculate_spectrum(signal, fs):
    """
    Calculate normalized FFT power in the plausible HR band.

    Returns the frequency axis, normalized power,
    bin-based HR, and interpolated HR.
    """
    signal = np.asarray(signal, dtype=float)

    signal = signal - np.mean(signal)
    signal = signal * np.hanning(len(signal))

    fft_result = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1.0 / fs
    )

    power = np.abs(fft_result) ** 2

    valid = (
        (frequencies >= LOW_FREQ)
        & (frequencies <= HIGH_FREQ)
    )

    valid_frequencies = frequencies[valid]
    valid_power = power[valid]

    if len(valid_power) == 0 or np.max(valid_power) == 0:
        return (
            valid_frequencies,
            valid_power,
            np.nan,
            np.nan
        )

    peak_local_index = np.argmax(valid_power)
    peak_global_index = np.flatnonzero(valid)[peak_local_index]

    bin_frequency = frequencies[peak_global_index]
    bin_hr = bin_frequency * 60.0

    # Refine the peak using parabolic interpolation of log power.
    interpolated_frequency = bin_frequency

    if 0 < peak_global_index < len(power) - 1:

        log_power = np.log(power + 1e-12)

        left = log_power[peak_global_index - 1]
        center = log_power[peak_global_index]
        right = log_power[peak_global_index + 1]

        denominator = left - 2 * center + right

        if abs(denominator) > 1e-12:

            offset = 0.5 * (
                left - right
            ) / denominator

            offset = np.clip(offset, -0.5, 0.5)

            interpolated_frequency = (
                peak_global_index + offset
            ) * fs / len(signal)

    interpolated_frequency = np.clip(
        interpolated_frequency,
        LOW_FREQ,
        HIGH_FREQ
    )

    interpolated_hr = interpolated_frequency * 60.0

    normalized_power = valid_power / np.max(valid_power)

    return (
        valid_frequencies,
        normalized_power,
        bin_hr,
        interpolated_hr
    )


# ============================================================
# 8. INSPECT THE SELECTED WINDOWS
# ============================================================

fig, axes = plt.subplots(
    len(PROBLEM_WINDOWS),
    3,
    figsize=(18, 12)
)

if len(PROBLEM_WINDOWS) == 1:
    axes = np.array([axes])

for row, (start_sec, end_sec) in enumerate(PROBLEM_WINDOWS):

    # --------------------------------------------------------
    # Extract camera rPPG segment
    # --------------------------------------------------------

    start_idx = round(start_sec * FPS)
    end_idx = round(end_sec * FPS)

    if end_idx > len(rppg):
        raise ValueError(
            f"Window {start_sec}-{end_sec}s exceeds video duration."
        )

    camera_segment = rppg[start_idx:end_idx]

    camera_time = (
        np.arange(start_idx, end_idx) / FPS
    )

    # --------------------------------------------------------
    # Extract reference PPG segment after full-recording filter
    # --------------------------------------------------------

    ppg_mask = (
        (uniform_time >= start_sec)
        & (uniform_time < end_sec)
    )

    reference_segment = filtered_reference_ppg[ppg_mask]
    reference_time = uniform_time[ppg_mask]

    # --------------------------------------------------------
    # Ground-truth HR values for the interval
    # --------------------------------------------------------

    hr_mask = (
        (gt_time >= start_sec)
        & (gt_time < end_sec)
    )

    window_gt_time = gt_time[hr_mask]
    window_gt_hr = gt_hr[hr_mask]

    if len(window_gt_hr) == 0:
        raise ValueError(
            f"No ground-truth HR samples in "
            f"{start_sec}-{end_sec}s."
        )

    gt_mean_hr = np.mean(window_gt_hr)

    # --------------------------------------------------------
    # Estimate dominant HR from both waveforms
    # --------------------------------------------------------

    (
        camera_f,
        camera_power,
        camera_bin_hr,
        camera_interp_hr
    ) = calculate_spectrum(camera_segment, FPS)

    (
        reference_f,
        reference_power,
        reference_bin_hr,
        reference_interp_hr
    ) = calculate_spectrum(reference_segment, GT_FS)

    print(
        f"\nWindow {start_sec}-{end_sec}s"
    )

    print(
        f"Ground-truth mean HR: {gt_mean_hr:.2f} BPM"
    )

    print(
        f"Camera rPPG HR: {camera_interp_hr:.2f} BPM"
    )

    print(
        f"Reference PPG HR: {reference_interp_hr:.2f} BPM"
    )

    # --------------------------------------------------------
    # Column 1: standardized waveform overlay
    # --------------------------------------------------------

    ax = axes[row, 0]

    ax.plot(
        camera_time,
        standardize(camera_segment),
        label="Camera rPPG"
    )

    ax.plot(
        reference_time,
        standardize(reference_segment),
        label="Reference PPG"
    )

    ax.set_title(
        f"{start_sec}-{end_sec}s: Waveforms"
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Standardized amplitude")
    ax.grid(True)
    ax.legend()

    # --------------------------------------------------------
    # Column 2: normalized power spectra
    # --------------------------------------------------------

    ax = axes[row, 1]

    ax.plot(
        camera_f,
        camera_power,
        label=f"Camera rPPG ({camera_interp_hr:.1f} BPM)"
    )

    ax.plot(
        reference_f,
        reference_power,
        label=f"Reference PPG ({reference_interp_hr:.1f} BPM)"
    )

    ax.axvline(
        gt_mean_hr / 60.0,
        linestyle="--",
        label=f"GT mean ({gt_mean_hr:.1f} BPM)"
    )

    ax.set_title(
        f"{start_sec}-{end_sec}s: Frequency spectra"
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized power")
    ax.grid(True)
    ax.legend(fontsize=8)

    # --------------------------------------------------------
    # Column 3: supplied HR trajectory
    # --------------------------------------------------------

    ax = axes[row, 2]

    ax.plot(
        window_gt_time,
        window_gt_hr,
        label="Supplied ground-truth HR"
    )

    ax.axhline(
        gt_mean_hr,
        linestyle="--",
        label=f"Window mean = {gt_mean_hr:.1f} BPM"
    )

    ax.set_title(
        f"{start_sec}-{end_sec}s: Reference HR values"
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("HR (BPM)")
    ax.grid(True)
    ax.legend(fontsize=8)

plt.suptitle(
    "Video 4: Investigation of High-Error HR Windows",
    fontsize=15
)

plt.tight_layout()

output_path = BASE_DIR / "vid4_problem_windows.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

print("\nSaved diagnostic plot:", output_path.resolve())

plt.show()