import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, detrend, find_peaks

# ============================================================
# 1. LOAD DATA
# ============================================================

rppg_df = pd.read_csv("filtered_rppg.csv")

gt = pd.read_csv("gtdump_2.xmp", header=None)
gt.columns = ["Timestamp_ms", "HR", "SpO2", "PPG"]

FPS = 28.671786
START_SEC = 0.0
END_SEC = 15.0

# Camera-derived rPPG segment
start_idx = round(START_SEC * FPS)
end_idx = round(END_SEC * FPS)

rppg = rppg_df["Filtered_rPPG"].to_numpy(dtype=float)[
    start_idx:end_idx
]

# Reference measurements for the same interval
gt["Time_sec"] = gt["Timestamp_ms"] / 1000.0

gt_window = gt[
    (gt["Time_sec"] >= START_SEC) &
    (gt["Time_sec"] < END_SEC)
].copy()

gt_time = gt_window["Time_sec"].to_numpy()
reference_ppg = gt_window["PPG"].to_numpy(dtype=float)

print("rPPG samples:", len(rppg))
print("Reference PPG samples:", len(reference_ppg))
print("Ground-truth mean HR:", gt_window["HR"].mean(), "BPM")

# ============================================================
# 2. PREPARE THE REFERENCE PPG
# ============================================================

# Estimate reference sampling frequency from timestamps
time_differences = np.diff(gt_time)
gt_fps = 1.0 / np.median(time_differences)

print("Estimated reference sampling rate:", gt_fps, "Hz")

# The reference timestamps may have small timing variations.
# Interpolate the PPG onto a uniformly sampled time grid.
uniform_time = np.arange(
    gt_time[0],
    gt_time[-1],
    1.0 / gt_fps
)

uniform_ppg = np.interp(
    uniform_time,
    gt_time,
    reference_ppg
)

# Remove slow baseline trend
uniform_ppg = detrend(uniform_ppg, type="linear")

# Band-pass the reference PPG using the same HR frequency band
sos = butter(
    4,
    [0.7, 4.0],
    btype="bandpass",
    fs=gt_fps,
    output="sos"
)

reference_filtered = sosfiltfilt(sos, uniform_ppg)

# ============================================================
# 3. FFT HELPER
# ============================================================

def calculate_spectrum(signal, sampling_rate):
    signal = signal - np.mean(signal)
    signal = signal * np.hanning(len(signal))

    fft_result = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1.0 / sampling_rate
    )

    power = np.abs(fft_result) ** 2

    valid = (frequencies >= 0.7) & (frequencies <= 4.0)

    frequencies = frequencies[valid]
    power = power[valid]

    return frequencies, power


# ============================================================
# 4. COMPARE FREQUENCY PEAKS
# ============================================================

r_freq, r_power = calculate_spectrum(rppg, FPS)

gt_freq, gt_power = calculate_spectrum(
    reference_filtered,
    gt_fps
)

r_peak = r_freq[np.argmax(r_power)]
gt_peak = gt_freq[np.argmax(gt_power)]

print("\n========== SPECTRAL COMPARISON ==========")
print("rPPG dominant frequency:", r_peak, "Hz")
print("rPPG dominant HR:", r_peak * 60, "BPM")

print("Reference PPG dominant frequency:", gt_peak, "Hz")
print("Reference PPG dominant HR:", gt_peak * 60, "BPM")

print("Ground-truth HR column mean:",
      gt_window["HR"].mean(), "BPM")
print("=========================================")

# ============================================================
# 5. PLOT BOTH NORMALIZED POWER SPECTRA
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    r_freq,
    r_power / np.max(r_power),
    label="Camera-derived rPPG"
)

plt.plot(
    gt_freq,
    gt_power / np.max(gt_power),
    label="Reference PPG"
)

plt.axvline(
    gt_window["HR"].mean() / 60,
    linestyle="--",
    label="Mean reference HR"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized power")
plt.title("First 15 Seconds: rPPG vs Reference PPG")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()