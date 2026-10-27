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

VIDEO_PATH = BASE_DIR / "vid_5.avi"
RPPG_PATH = BASE_DIR / "filtered_rppg.csv"
GT_PATH = BASE_DIR / "gtdump_5.xmp"

LOW_FREQ = 0.7
HIGH_FREQ = 4.0

HR_WINDOW_SECONDS = 15
HOP_SECONDS = 5


# ============================================================
# 2. VERIFY FILES
# ============================================================

for path in [VIDEO_PATH, RPPG_PATH, GT_PATH]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path.resolve()}"
        )

print("========== INPUT FILES ==========")
print("Video:", VIDEO_PATH.resolve())
print("Filtered rPPG:", RPPG_PATH.resolve())
print("Ground truth:", GT_PATH.resolve())


# ============================================================
# 3. READ VIDEO METADATA
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

FPS = cap.get(cv2.CAP_PROP_FPS)
VIDEO_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()

if FPS <= 0 or VIDEO_FRAMES <= 0:
    raise ValueError("Invalid video FPS or frame count.")

VIDEO_DURATION = VIDEO_FRAMES / FPS

print("\n========== VIDEO METADATA ==========")
print("FPS:", FPS)
print("Frame count:", VIDEO_FRAMES)
print("Duration:", VIDEO_DURATION, "seconds")


# ============================================================
# 4. LOAD CAMERA rPPG
# ============================================================

rppg_df = pd.read_csv(RPPG_PATH)

if "Filtered_rPPG" not in rppg_df.columns:
    raise ValueError(
        "filtered_rppg.csv must contain 'Filtered_rPPG'."
    )

rppg = rppg_df["Filtered_rPPG"].to_numpy(
    dtype=np.float64
)

if len(rppg) != VIDEO_FRAMES:
    raise ValueError(
        f"Video has {VIDEO_FRAMES} frames but the rPPG "
        f"file has {len(rppg)} samples. Check that these "
        f"files belong to Video 4."
    )

if not np.all(np.isfinite(rppg)):
    raise ValueError("rPPG contains NaN or infinite values.")

print("\n========== CAMERA rPPG ==========")
print("Samples:", len(rppg))
print("Duration:", len(rppg) / FPS, "seconds")


# ============================================================
# 5. LOAD VIDEO 4 GROUND TRUTH
# ============================================================

gt = pd.read_csv(GT_PATH, header=None)

if gt.shape[1] != 4:
    raise ValueError(
        f"Expected 4 ground-truth columns; got {gt.shape[1]}."
    )

gt.columns = [
    "Timestamp_ms",
    "HR",
    "SpO2",
    "PPG"
]

gt["Time_sec"] = gt["Timestamp_ms"] / 1000.0

gt_time = gt["Time_sec"].to_numpy(dtype=np.float64)
gt_hr = gt["HR"].to_numpy(dtype=np.float64)
gt_ppg = gt["PPG"].to_numpy(dtype=np.float64)

if not (
    np.all(np.isfinite(gt_time))
    and np.all(np.isfinite(gt_hr))
    and np.all(np.isfinite(gt_ppg))
):
    raise ValueError("Ground-truth data contains invalid values.")

if np.any(np.diff(gt_time) <= 0):
    raise ValueError(
        "Ground-truth timestamps must increase strictly."
    )

GT_FS = 1.0 / np.median(np.diff(gt_time))
GT_DURATION = gt_time[-1] - gt_time[0]

print("\n========== GROUND-TRUTH METADATA ==========")
print("File:", GT_PATH.resolve())
print("Rows:", len(gt))
print("Reference PPG sampling rate:", GT_FS, "Hz")
print("First timestamp:", gt_time[0], "seconds")
print("Last timestamp:", gt_time[-1], "seconds")
print("Ground-truth duration:", GT_DURATION, "seconds")
print("Mean HR:", gt_hr.mean(), "BPM")
print("Median HR:", np.median(gt_hr), "BPM")
print("Minimum HR:", gt_hr.min(), "BPM")
print("Maximum HR:", gt_hr.max(), "BPM")


# ============================================================
# 6. PREPARE THE COMPLETE REFERENCE PPG
# ============================================================

# Create a uniform time grid using the reference sampling rate.
uniform_time = np.arange(
    gt_time[0],
    gt_time[-1],
    1.0 / GT_FS
)

# Interpolate the reference waveform onto that grid.
uniform_ppg = np.interp(
    uniform_time,
    gt_time,
    gt_ppg
)

# Remove a linear baseline trend from the complete recording.
uniform_ppg = detrend(
    uniform_ppg,
    type="linear"
)

# Filter the full reference waveform BEFORE extracting windows.
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
# 7. HR ESTIMATION FUNCTION
# ============================================================

def estimate_hr(signal, sampling_rate):
    """
    Return:
        bin_hr: HR from the strongest FFT bin
        interpolated_hr: HR after parabolic peak interpolation
    """

    signal = np.asarray(signal, dtype=np.float64)

    if len(signal) < 3:
        return np.nan, np.nan

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal window contains invalid values.")

    # Remove the mean and reduce spectral leakage.
    signal = signal - np.mean(signal)
    signal = signal * np.hanning(len(signal))

    spectrum = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1.0 / sampling_rate
    )

    power = np.abs(spectrum) ** 2

    # Restrict peak detection to 0.7–4.0 Hz.
    valid_indices = np.flatnonzero(
        (frequencies >= LOW_FREQ)
        & (frequencies <= HIGH_FREQ)
    )

    if len(valid_indices) == 0:
        return np.nan, np.nan

    peak_index = valid_indices[
        np.argmax(power[valid_indices])
    ]

    peak_frequency = frequencies[peak_index]
    bin_hr = peak_frequency * 60.0

    # Parabolic interpolation using neighboring log-power bins.
    interpolated_frequency = peak_frequency

    if 0 < peak_index < len(power) - 1:
        log_power = np.log(power + 1e-12)

        left = log_power[peak_index - 1]
        center = log_power[peak_index]
        right = log_power[peak_index + 1]

        denominator = left - 2.0 * center + right

        if abs(denominator) > 1e-12:
            offset = 0.5 * (left - right) / denominator
            offset = np.clip(offset, -0.5, 0.5)

            interpolated_frequency = (
                peak_index + offset
            ) * sampling_rate / len(signal)

    interpolated_frequency = np.clip(
        interpolated_frequency,
        LOW_FREQ,
        HIGH_FREQ
    )

    interpolated_hr = interpolated_frequency * 60.0

    return bin_hr, interpolated_hr


# ============================================================
# 8. THREE-SIGNAL COMPARISON
# ============================================================

# Only evaluate complete 15-second windows that fit within
# both recordings.
COMMON_DURATION = min(
    VIDEO_DURATION,
    GT_DURATION
)

last_start = COMMON_DURATION - HR_WINDOW_SECONDS

if last_start < 0:
    raise ValueError("Recording is shorter than 15 seconds.")

window_starts = np.arange(
    0,
    last_start + 1e-9,
    HOP_SECONDS
)

results = []

for start_sec in window_starts:

    end_sec = start_sec + HR_WINDOW_SECONDS

    # --------------------------------------------------------
    # Camera-derived rPPG
    # --------------------------------------------------------

    start_idx = round(start_sec * FPS)
    end_idx = round(end_sec * FPS)

    rppg_segment = rppg[start_idx:end_idx]

    camera_bin_hr, camera_interp_hr = estimate_hr(
        rppg_segment,
        FPS
    )

    # --------------------------------------------------------
    # Reference PPG
    # --------------------------------------------------------

    ppg_mask = (
        (uniform_time >= start_sec)
        & (uniform_time < end_sec)
    )

    ppg_segment = filtered_reference_ppg[ppg_mask]

    ppg_bin_hr, ppg_interp_hr = estimate_hr(
        ppg_segment,
        GT_FS
    )

    # --------------------------------------------------------
    # Supplied HR column
    # --------------------------------------------------------

    hr_mask = (
        (gt_time >= start_sec)
        & (gt_time < end_sec)
    )

    gt_window_hr = gt_hr[hr_mask]

    if len(gt_window_hr) == 0:
        continue

    gt_mean_hr = np.mean(gt_window_hr)
    gt_median_hr = np.median(gt_window_hr)

    results.append({
        "Start_sec": start_sec,
        "End_sec": end_sec,
        "Center_sec": (start_sec + end_sec) / 2.0,
        "Camera_HR_Bin": camera_bin_hr,
        "Camera_HR_Interpolated": camera_interp_hr,
        "Reference_PPG_HR_Bin": ppg_bin_hr,
        "Reference_PPG_HR_Interpolated": ppg_interp_hr,
        "GT_Mean_HR": gt_mean_hr,
        "GT_Median_HR": gt_median_hr,
        "Camera_vs_PPG_Error": abs(
            camera_interp_hr - ppg_interp_hr
        ),
        "Camera_vs_HRColumn_Error": abs(
            camera_interp_hr - gt_mean_hr
        ),
        "PPG_vs_HRColumn_Error": abs(
            ppg_interp_hr - gt_mean_hr
        )
    })


# ============================================================
# 9. DISPLAY AND SAVE RESULTS
# ============================================================

result_df = pd.DataFrame(results)

if result_df.empty:
    raise RuntimeError("No valid HR windows were evaluated.")

print("\n========== VIDEO 4: THREE-SIGNAL COMPARISON ==========")
print(result_df.round(3).to_string(index=False))

output_path = BASE_DIR / "three_signal_comparison_vid4.csv"

result_df.to_csv(
    output_path,
    index=False
)

print("\nSaved:", output_path.resolve())


# ============================================================
# 10. SUMMARY METRICS
# ============================================================

print("\n========== SUMMARY METRICS ==========")

comparisons = [
    ("Camera rPPG vs Reference PPG",
     "Camera_vs_PPG_Error"),
    ("Camera rPPG vs Supplied HR",
     "Camera_vs_HRColumn_Error"),
    ("Reference PPG vs Supplied HR",
     "PPG_vs_HRColumn_Error")
]

for label, column in comparisons:
    errors = result_df[column].to_numpy(dtype=float)

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    print(f"\n{label}")
    print(f"MAE:  {mae:.3f} BPM")
    print(f"RMSE: {rmse:.3f} BPM")

print("======================================")


# ============================================================
# 11. PLOT ALL THREE ESTIMATES
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
plt.title("Video 4: Camera rPPG vs Reference PPG vs Ground Truth")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()