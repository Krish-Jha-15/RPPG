import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, find_peaks

# ============================================================
# 1. LOAD THE MATCHING VIDEO 2 GROUND-TRUTH FILE
# ============================================================

gt = pd.read_csv("gtdump_2.xmp", header=None)

gt.columns = ["Timestamp_ms", "HR", "SpO2", "PPG"]

gt["Time_sec"] = gt["Timestamp_ms"] / 1000.0

# First 15 seconds
segment = gt[
    (gt["Time_sec"] >= 0) &
    (gt["Time_sec"] < 15)
].copy()

t = segment["Time_sec"].to_numpy()
hr = segment["HR"].to_numpy(dtype=float)
ppg = segment["PPG"].to_numpy(dtype=float)

# Estimate PPG sampling rate from timestamps
fs = 1.0 / np.median(np.diff(t))

print("Samples:", len(segment))
print("Estimated PPG sampling rate:", fs, "Hz")

print("\nGround-truth HR statistics:")
print("Mean:", np.mean(hr), "BPM")
print("Median:", np.median(hr), "BPM")
print("Minimum:", np.min(hr), "BPM")
print("Maximum:", np.max(hr), "BPM")

# ============================================================
# 2. FILTER THE REFERENCE PPG FOR VISUAL INSPECTION
# ============================================================

sos = butter(
    4,
    [0.7, 4.0],
    btype="bandpass",
    fs=fs,
    output="sos"
)

ppg_filtered = sosfiltfilt(sos, ppg)

# ============================================================
# 3. OPTIONAL ROUGH PULSE-PEAK CHECK
# ============================================================

# This is diagnostic only. Automatic peak detection can fail
# if the waveform contains noise, multiple peaks per pulse,
# or inverted pulse shapes.
minimum_distance = max(1, int(fs * 60 / 180))

peaks, _ = find_peaks(
    ppg_filtered,
    distance=minimum_distance,
    prominence=0.2 * np.std(ppg_filtered)
)

if len(peaks) >= 2:
    intervals = np.diff(t[peaks])

    rough_peak_hr = 60.0 / np.median(intervals)

    print("\nRough positive-peak estimate:",
          rough_peak_hr, "BPM")
    print("Detected positive peaks:", len(peaks))
else:
    print("\nToo few positive peaks for a rough estimate.")

# ============================================================
# INSPECT DETECTED PPG PEAKS
# ============================================================

if len(peaks) >= 2:
    peak_times = t[peaks]
    intervals = np.diff(peak_times)

    print("\n========== PPG PEAK INTERVALS ==========")

    print("Detected peaks:", len(peaks))
    print("Median interval:", np.median(intervals), "seconds")
    print("Mean interval:", np.mean(intervals), "seconds")
    print("Median-interval HR:",
          60.0 / np.median(intervals), "BPM")

    print("Expected interval at mean GT HR:",
          60.0 / np.mean(hr), "seconds")

    print("\nFirst 10 peak intervals:")
    print(intervals[:10])

    print("========================================")

# ============================================================
# PLOT THE FIRST FIVE SECONDS
# ============================================================

first_five_seconds = t < 5
peaks_first_five = peaks[t[peaks] < 5]

plt.figure(figsize=(12, 5))

plt.plot(
    t[first_five_seconds],
    ppg_filtered[first_five_seconds],
    label="Filtered reference PPG"
)

plt.scatter(
    t[peaks_first_five],
    ppg_filtered[peaks_first_five],
    marker="x",
    s=70,
    label="Detected peaks"
)

plt.xlabel("Time (seconds)")
plt.ylabel("PPG amplitude")
plt.title("Video 2: Reference PPG Peaks, First Five Seconds")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 4. PLOT THE GROUND-TRUTH HR COLUMN
# ============================================================

plt.figure(figsize=(12, 4))

plt.plot(t, hr)

plt.xlabel("Time (seconds)")
plt.ylabel("Ground-truth HR (BPM)")
plt.title("Video 2: Ground-Truth HR, First 15 Seconds")
plt.grid(True)
plt.tight_layout()

# ============================================================
# 5. PLOT REFERENCE PPG
# ============================================================

plt.figure(figsize=(12, 4))

plt.plot(t, ppg, label="Reference PPG")
plt.plot(t, ppg_filtered, label="Band-pass filtered PPG")

if len(peaks) > 0:
    plt.scatter(
        t[peaks],
        ppg_filtered[peaks],
        label="Detected positive peaks",
        s=20
    )

plt.xlabel("Time (seconds)")
plt.ylabel("PPG amplitude")
plt.title("Video 2: Reference PPG, First 15 Seconds")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
