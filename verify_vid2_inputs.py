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

VIDEO_PATH = BASE_DIR / "vid2.avi"
FILTERED_RPPG_PATH = BASE_DIR / "filtered_rppg.csv"
GROUND_TRUTH_PATH = BASE_DIR / "gtdump_2.xmp"

LOW_FREQ = 0.7
HIGH_FREQ = 4.0

HR_WINDOW_SECONDS = 15
HOP_SECONDS = 5


# ============================================================
# 2. VERIFY INPUT FILES
# ============================================================

for path in [
    VIDEO_PATH,
    FILTERED_RPPG_PATH,
    GROUND_TRUTH_PATH
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

print("========== INPUT FILES ==========")
print("Video:", VIDEO_PATH.resolve())
print("Filtered rPPG:", FILTERED_RPPG_PATH.resolve())
print("Ground truth:", GROUND_TRUTH_PATH.resolve())


# ============================================================
# 3. READ VIDEO METADATA
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

FPS = cap.get(cv2.CAP_PROP_FPS)
VIDEO_FRAME_COUNT = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

cap.release()

if FPS <= 0:
    raise ValueError("Invalid video FPS.")

VIDEO_DURATION = VIDEO_FRAME_COUNT / FPS

print("\n========== VIDEO METADATA ==========")
print("FPS:", FPS)
print("Video frames:", VIDEO_FRAME_COUNT)
print("Video duration:", VIDEO_DURATION, "seconds")


# ============================================================
# 4. LOAD FILTERED CAMERA rPPG
# ============================================================

rppg_df = pd.read_csv(FILTERED_RPPG_PATH)

if "Filtered_rPPG" not in rppg_df.columns:
    raise ValueError(
        "filtered_rppg.csv must contain 'Filtered_rPPG'."
    )

rppg = rppg_df[
    "Filtered_rPPG"
].to_numpy(dtype=np.float64)

if not np.all(np.isfinite(rppg)):
    raise ValueError(
        "Filtered rPPG contains NaN or infinite values."
    )

if len(rppg) != VIDEO_FRAME_COUNT:
    raise ValueError(
        f"Video has {VIDEO_FRAME_COUNT} frames, "
        f"but filtered rPPG has {len(rppg)} samples. "
        "Verify that both files belong to Video 2."
    )

print("\n========== CAMERA rPPG ==========")
print("Samples:", len(rppg))
print("Duration:", len(rppg) / FPS, "seconds")


# ============================================================
# 5. LOAD THE CORRECT VIDEO 2 GROUND TRUTH
# ============================================================

gt = pd.read_csv(
    GROUND_TRUTH_PATH,
    header=None
)

if gt.shape[1] != 4:
    raise ValueError(
        f"Expected four ground-truth columns; got {gt.shape[1]}."
    )

gt.columns = [
    "Timestamp_ms",
    "HR",
    "SpO2",
    "PPG"
]

gt["Time_sec"] = (
    gt["Timestamp_ms"].to_numpy(dtype=np.float64)
    / 1000.0
)

gt_time = gt["Time_sec"].to_numpy(dtype=np.float64)
gt_hr = gt["HR"].to_numpy(dtype=np.float64)
gt_ppg = gt["PPG"].to_numpy(dtype=np.float64)

if not np.all(np.isfinite(gt_time)):
    raise ValueError("Ground-truth timestamps contain invalid values.")

if not np.all(np.isfinite(gt_hr)):
    raise ValueError("Ground-truth HR contains invalid values.")

if not np.all(np.isfinite(gt_ppg)):
    raise ValueError("Ground-truth PPG contains invalid values.")

if np.any(np.diff(gt_time) <= 0):
    raise ValueError(
        "Ground-truth timestamps must be strictly increasing."
    )

GT_FS = 1.0 / np.median(np.diff(gt_time))

GT_DURATION = gt_time[-1] - gt_time[0]

print("\n========== GROUND-TRUTH METADATA ==========")
print("File:", GROUND_TRUTH_PATH.resolve())
print("Rows:", len(gt))
print("Estimated PPG sampling rate:", GT_FS, "Hz")
print("First timestamp:", gt_time[0], "seconds")
print("Last timestamp:", gt_time[-1], "seconds")
print("Ground-truth duration:", GT_DURATION, "seconds")
print("Overall mean HR:", np.mean(gt_hr), "BPM")
print("Overall median HR:", np.median(gt_hr), "BPM")


# ============================================================
# 6. VERIFY THE FIRST 15-SECOND REFERENCE INTERVAL
# ============================================================

first_window_mask = (
    (gt_time >= 0.0) &
    (gt_time < HR_WINDOW_SECONDS)
)

first_window_gt_hr = gt_hr[first_window_mask]

if len(first_window_gt_hr) == 0:
    raise ValueError(
        "No ground-truth HR samples in the first window."
    )

print("\n========== FIRST WINDOW VERIFICATION ==========")
print("Interval: 0–15 seconds")
print("Reference samples:", len(first_window_gt_hr))
print("Mean reference HR:", np.mean(first_window_gt_hr), "BPM")
print("Median reference HR:", np.median(first_window_gt_hr), "BPM")


# ============================================================
# 7. PREPARE THE COMPLETE REFERENCE PPG
# ============================================================

# Timestamps can have small sampling variations.
# Interpolate the entire PPG onto a uniform time grid.

uniform_time = np.arange(
    gt_time[0],
    gt_time[-1],
    1.0 / GT_FS
)

uniform_ppg = np.interp(
    uniform_time,
    gt_time,
    gt_ppg
)

# Remove a linear baseline trend over the entire recording.
uniform_ppg = detrend(
    uniform_ppg,
    type="linear"
)

# Filter the FULL reference recording before extracting
# individual windows. This reduces segment-edge effects.

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
# 8. FUNCTION TO ESTIMATE DOMINANT HR
# ============================================================

def estimate_dominant_hr(signal, sampling_rate):
    """
    Estimate HR using the strongest FFT power bin
    and parabolic interpolation of log power.

    Returns:
        bin_hr: frequency-bin HR in BPM
        interpolated_hr: interpolated HR in BPM
    """

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    if len(signal) < 3:
        return np.nan, np.nan

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal window contains invalid values.")

    # Remove the segment mean.
    signal = signal - np.mean(signal)

    # Reduce spectral leakage.
    signal = signal * np.hanning(len(signal))

    # Real FFT.
    fft_result = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1.0 / sampling_rate
    )

    power = np.abs(fft_result) ** 2

    # Only search frequencies corresponding to 42–240 BPM.
    valid_indices = np.flatnonzero(
        (frequencies >= LOW_FREQ) &
        (frequencies <= HIGH_FREQ)
    )

    if len(valid_indices) == 0:
        return np.nan, np.nan

    # Strongest power bin within the HR frequency range.
    peak_index = valid_indices[
        np.argmax(power[valid_indices])
    ]

    peak_frequency = frequencies[peak_index]

    # Frequency-bin estimate.
    bin_hr = peak_frequency * 60.0

    # Interpolate peak position using neighboring log-power bins.
    interpolated_frequency = peak_frequency

    if 0 < peak_index < len(power) - 1:

        log_power = np.log(power + 1e-12)

        left = log_power[peak_index - 1]
        center = log_power[peak_index]
        right = log_power[peak_index + 1]

        denominator = (
            left - 2.0 * center + right
        )

        if abs(denominator) > 1e-12:

            offset = 0.5 * (
                left - right
            ) / denominator

            offset = np.clip(
                offset,
                -0.5,
                0.5
            )

            interpolated_frequency = (
                peak_index + offset
            ) * sampling_rate / len(signal)

    # Keep the interpolated frequency inside the requested band.
    interpolated_frequency = np.clip(
        interpolated_frequency,
        LOW_FREQ,
        HIGH_FREQ
    )

    interpolated_hr = (
        interpolated_frequency * 60.0
    )

    return bin_hr, interpolated_hr


# ============================================================
# 9. THREE-SIGNAL COMPARISON
# ============================================================

# Only use windows fully covered by both recordings.
common_duration = min(
    VIDEO_DURATION,
    GT_DURATION
)

results = []

latest_start = (
    common_duration - HR_WINDOW_SECONDS
)

if latest_start < 0:
    raise ValueError(
        "Recording is shorter than the HR-analysis window."
    )

window_starts = np.arange(
    0.0,
    latest_start + 1e-9,
    HOP_SECONDS
)

for start_sec in window_starts:

    end_sec = start_sec + HR_WINDOW_SECONDS

    # --------------------------------------------------------
    # Camera-derived rPPG
    # --------------------------------------------------------

    r_start = round(start_sec * FPS)
    r_end = round(end_sec * FPS)

    r_segment = rppg[r_start:r_end]

    camera_bin_hr, camera_interpolated_hr = (
        estimate_dominant_hr(
            r_segment,
            FPS
        )
    )

    # --------------------------------------------------------
    # Reference PPG
    # --------------------------------------------------------

    ppg_mask = (
        (uniform_time >= start_sec) &
        (uniform_time < end_sec)
    )

    ppg_segment = filtered_reference_ppg[ppg_mask]

    ppg_bin_hr, ppg_interpolated_hr = (
        estimate_dominant_hr(
            ppg_segment,
            GT_FS
        )
    )

    # --------------------------------------------------------
    # Supplied ground-truth HR column
    # --------------------------------------------------------

    hr_mask = (
        (gt_time >= start_sec) &
        (gt_time < end_sec)
    )

    gt_window_hr = gt_hr[hr_mask]

    if len(gt_window_hr) == 0:
        continue

    gt_mean_hr = np.mean(gt_window_hr)
    gt_median_hr = np.median(gt_window_hr)

    # --------------------------------------------------------
    # Errors based on interpolated waveform estimates
    # --------------------------------------------------------

    camera_vs_ppg_error = abs(
        camera_interpolated_hr -
        ppg_interpolated_hr
    )

    camera_vs_hr_error = abs(
        camera_interpolated_hr -
        gt_mean_hr
    )

    ppg_vs_hr_error = abs(
        ppg_interpolated_hr -
        gt_mean_hr
    )

    results.append({
        "Start_sec": start_sec,
        "End_sec": end_sec,
        "Center_sec": (
            start_sec + end_sec
        ) / 2.0,
        "Camera_HR_Bin": camera_bin_hr,
        "Camera_HR_Interpolated": camera_interpolated_hr,
        "Reference_PPG_HR_Bin": ppg_bin_hr,
        "Reference_PPG_HR_Interpolated": ppg_interpolated_hr,
        "GT_Mean_HR": gt_mean_hr,
        "GT_Median_HR": gt_median_hr,
        "Camera_vs_PPG_Error": camera_vs_ppg_error,
        "Camera_vs_HRColumn_Error": camera_vs_hr_error,
        "PPG_vs_HRColumn_Error": ppg_vs_hr_error
    })


# ============================================================
# 10. DISPLAY AND SAVE THE RESULTS
# ============================================================

result_df = pd.DataFrame(results)

if result_df.empty:
    raise RuntimeError(
        "No HR windows could be evaluated."
    )

print("\n========== THREE-SIGNAL COMPARISON ==========")

print(
    result_df.round(3).to_string(
        index=False
    )
)

output_path = BASE_DIR / "three_signal_comparison_vid2.csv"

result_df.to_csv(
    output_path,
    index=False
)

print("\nSaved comparison:", output_path.resolve())


# ============================================================
# 11. SUMMARY METRICS
# ============================================================

print("\n========== SUMMARY METRICS ==========")

metrics = [
    (
        "Camera rPPG vs Reference PPG",
        "Camera_vs_PPG_Error"
    ),
    (
        "Camera rPPG vs Supplied HR Column",
        "Camera_vs_HRColumn_Error"
    ),
    (
        "Reference PPG vs Supplied HR Column",
        "PPG_vs_HRColumn_Error"
    )
]

for description, column in metrics:

    errors = result_df[column].to_numpy(
        dtype=np.float64
    )

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    print(f"\n{description}")
    print(f"MAE:  {mae:.3f} BPM")
    print(f"RMSE: {rmse:.3f} BPM")

print("\n======================================")


# ============================================================
# 12. PLOT ALL THREE HR ESTIMATES
# ============================================================

plt.figure(figsize=(13, 6))

plt.plot(
    result_df["Center_sec"],
    result_df["Camera_HR_Interpolated"],
    marker="o",
    label="Camera-derived rPPG"
)

plt.plot(
    result_df["Center_sec"],
    result_df["Reference_PPG_HR_Interpolated"],
    marker="s",
    label="Reference PPG spectrum"
)

plt.plot(
    result_df["Center_sec"],
    result_df["GT_Mean_HR"],
    marker="^",
    label="Supplied ground-truth HR mean"
)

plt.xlabel("Window center time (seconds)")
plt.ylabel("Heart Rate (BPM)")

plt.title(
    "Video 2: Camera rPPG vs Reference PPG vs Ground Truth"
)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
