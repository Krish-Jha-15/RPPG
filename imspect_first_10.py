
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ============================================================
# 1. LOAD VIDEO 2 FILTERED rPPG
# ============================================================

df = pd.read_csv("filtered_rppg.csv")

signal = df["Filtered_rPPG"].to_numpy(dtype=np.float64)

FPS = 28.671786
WINDOW_SECONDS = 15

# Extract the first 15 seconds
start_idx = 0
end_idx = round(WINDOW_SECONDS * FPS)

segment = signal[start_idx:end_idx]

print("Number of samples:", len(segment))
print("Expected duration:", len(segment) / FPS, "seconds")

# ============================================================
# 2. LOAD MATCHING GROUND TRUTH
# ============================================================

gt = pd.read_csv("gtdump_2.xmp", header=None)
gt.columns = ["Timestamp_ms", "HR", "SpO2", "PPG"]

gt["Time_sec"] = gt["Timestamp_ms"] / 1000.0

gt_first_window = gt[
    (gt["Time_sec"] >= 0) &
    (gt["Time_sec"] < WINDOW_SECONDS)
]

print("\nGround-truth mean HR:",
      gt_first_window["HR"].mean(), "BPM")

print("Ground-truth median HR:",
      gt_first_window["HR"].median(), "BPM")

# ============================================================
# 3. REMOVE MEAN AND APPLY HANN WINDOW
# ============================================================

segment = segment - np.mean(segment)

hann = np.hanning(len(segment))
windowed_segment = segment * hann

# ============================================================
# 4. COMPUTE FFT AND POWER SPECTRUM
# ============================================================

fft_result = np.fft.rfft(windowed_segment)

frequencies = np.fft.rfftfreq(
    len(windowed_segment),
    d=1 / FPS
)

power = np.abs(fft_result) ** 2

LOW_FREQ = 0.7
HIGH_FREQ = 4.0

valid_mask = (
    (frequencies >= LOW_FREQ) &
    (frequencies <= HIGH_FREQ)
)

valid_indices = np.flatnonzero(valid_mask)
valid_power = power[valid_indices]

# Find local maxima in the valid HR range
peak_indices, _ = find_peaks(valid_power)

if len(peak_indices) == 0:
    peak_indices = np.array([np.argmax(valid_power)])

# Rank peaks by power
peak_indices = peak_indices[
    np.argsort(valid_power[peak_indices])[::-1]
]

print("\n========== STRONGEST FFT PEAKS ==========")

for rank, peak_idx in enumerate(peak_indices[:5], start=1):
    fft_idx = valid_indices[peak_idx]
    frequency = frequencies[fft_idx]
    hr = frequency * 60

    print(
        f"Peak {rank}: "
        f"{frequency:.4f} Hz, "
        f"{hr:.2f} BPM, "
        f"power={power[fft_idx]:.6g}"
    )

print("=========================================")

# ============================================================
# 5. PLOT THE FIRST 15 SECONDS OF FILTERED rPPG
# ============================================================

time = np.arange(len(segment)) / FPS

plt.figure(figsize=(12, 4))
plt.plot(time, segment)
plt.xlabel("Time (seconds)")
plt.ylabel("Filtered rPPG amplitude")
plt.title("Video 2: Filtered rPPG, First 15 Seconds")
plt.grid(True)
plt.tight_layout()

# ============================================================
# 6. PLOT THE POWER SPECTRUM
# ============================================================

plt.figure(figsize=(12, 4))

valid_frequencies = frequencies[valid_mask]
plt.plot(valid_frequencies, valid_power)

# Mark the strongest peak
strongest_idx = valid_indices[peak_indices[0]]
strongest_frequency = frequencies[strongest_idx]

plt.axvline(
    strongest_frequency,
    linestyle="--",
    label=f"Strongest peak: {strongest_frequency * 60:.2f} BPM"
)

# Reference HR mean for the first 15 seconds
gt_mean = gt_first_window["HR"].mean()
plt.axvline(
    gt_mean / 60,
    linestyle=":",
    label=f"GT mean: {gt_mean:.2f} BPM"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("Video 2: First 15-Second FFT Power Spectrum")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
