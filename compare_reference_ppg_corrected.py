import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, detrend

# ============================================================
# 1. LOAD THE COMPLETE VIDEO 2 GROUND-TRUTH FILE
# ============================================================

gt = pd.read_csv("gtdump_2.xmp", header=None)

gt.columns = ["Timestamp_ms", "HR", "SpO2", "PPG"]

gt_time = gt["Timestamp_ms"].to_numpy(dtype=float) / 1000.0
gt_ppg = gt["PPG"].to_numpy(dtype=float)

# Estimate sampling frequency from timestamps
gt_fps = 1.0 / np.median(np.diff(gt_time))

print("Ground-truth samples:", len(gt))
print("Estimated sampling rate:", gt_fps, "Hz")
print("Reference duration:",
      gt_time[-1] - gt_time[0], "seconds")

# ============================================================
# 2. RESAMPLE THE ENTIRE REFERENCE PPG TO A UNIFORM GRID
# ============================================================

uniform_time = np.arange(
    gt_time[0],
    gt_time[-1],
    1.0 / gt_fps
)

uniform_ppg = np.interp(
    uniform_time,
    gt_time,
    gt_ppg
)

# Remove a slow overall trend
uniform_ppg = detrend(uniform_ppg, type="linear")

# ============================================================
# 3. FILTER THE COMPLETE PPG SIGNAL
# ============================================================

sos = butter(
    4,
    [0.7, 4.0],
    btype="bandpass",
    fs=gt_fps,
    output="sos"
)

# Filter the full recording, not only a 15-second segment
reference_filtered = sosfiltfilt(
    sos,
    uniform_ppg
)

# ============================================================
# 4. EXTRACT THE FIRST 15 SECONDS AFTER FILTERING
# ============================================================

START_SEC = 0.0
END_SEC = 15.0

mask = (
    (uniform_time >= START_SEC) &
    (uniform_time < END_SEC)
)

reference_segment = reference_filtered[mask]
reference_time = uniform_time[mask]

# Load the camera-derived rPPG
rppg_df = pd.read_csv("filtered_rppg.csv")

FPS = 28.671786

rppg = rppg_df["Filtered_rPPG"].to_numpy(dtype=float)

start_idx = round(START_SEC * FPS)
end_idx = round(END_SEC * FPS)

rppg_segment = rppg[start_idx:end_idx]

print("\nReference PPG segment:", len(reference_segment))
print("Camera rPPG segment:", len(rppg_segment))

# Ground-truth HR for the same time period
gt_window = gt[
    (gt_time >= START_SEC) &
    (gt_time < END_SEC)
]

print("Mean ground-truth HR:",
      gt_window["HR"].mean(), "BPM")

# ============================================================
# 5. CALCULATE THE DOMINANT FREQUENCY
# ============================================================

def dominant_frequency(signal, fs):
    signal = signal - np.mean(signal)
    signal = signal * np.hanning(len(signal))

    spectrum = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1.0 / fs
    )

    power = np.abs(spectrum) ** 2

    valid = (
        (frequencies >= 0.7) &
        (frequencies <= 4.0)
    )

    frequencies = frequencies[valid]
    power = power[valid]

    peak_index = np.argmax(power)

    return frequencies[peak_index], frequencies, power

r_freq, r_frequencies, r_power = dominant_frequency(
    rppg_segment, FPS
)

gt_freq, gt_frequencies, gt_power = dominant_frequency(
    reference_segment, gt_fps
)

print("\n========== CORRECTED COMPARISON ==========")

print("Camera rPPG frequency:", r_freq, "Hz")
print("Camera rPPG HR:", r_freq * 60, "BPM")

print("Reference PPG frequency:", gt_freq, "Hz")
print("Reference PPG HR:", gt_freq * 60, "BPM")

print("Difference in waveform estimates:",
      abs(r_freq - gt_freq) * 60, "BPM")

print("==========================================")

# ============================================================
# 6. PLOT NORMALIZED SPECTRA
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    r_frequencies,
    r_power / np.max(r_power),
    label="Camera-derived rPPG"
)

plt.plot(
    gt_frequencies,
    gt_power / np.max(gt_power),
    label="Reference PPG (filtered over full recording)"
)

plt.axvline(
    gt_window["HR"].mean() / 60,
    linestyle="--",
    label="Mean ground-truth HR frequency"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized power")
plt.title("First 15 Seconds: Corrected rPPG vs Reference Spectrum")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
