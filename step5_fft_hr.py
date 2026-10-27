import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def estimate_heart_rate(
    input_file,
    # FPS=28.671786,
    # FPS=29.264106,
    FPS,
    LOW_HR=42,
    HIGH_HR=240
):

    # Read filtered rPPG signal
    df = pd.read_csv(input_file)

    signal = df["Filtered_rPPG"].to_numpy(dtype=np.float64)

    print("Sampling frequency:", FPS)
    print("Number of samples:", len(signal))

    if len(signal) < 2:
        raise ValueError("Signal must contain at least 2 samples.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or infinite values.")

    duration = len(signal) / FPS
    print("Signal duration:", duration, "seconds")

    # Remove DC component
    signal = signal - np.mean(signal)

    # Apply Hann window
    window = np.hanning(len(signal))
    windowed_signal = signal * window

    # FFT
    N = len(windowed_signal)

    fft_result = np.fft.rfft(windowed_signal)

    frequencies = np.fft.rfftfreq(
        N,
        d=1 / FPS
    )

    power = np.abs(fft_result) ** 2

    # Convert BPM limits to Hz
    LOW_FREQ = LOW_HR / 60
    HIGH_FREQ = HIGH_HR / 60

    # Select valid heart-rate frequencies
    valid = (
        (frequencies >= LOW_FREQ) &
        (frequencies <= HIGH_FREQ)
    )

    valid_frequencies = frequencies[valid]
    valid_power = power[valid]

    if len(valid_power) == 0:
        raise ValueError(
            "No FFT frequencies found in the selected heart-rate range."
        )

    # Find dominant frequency
    peak_index = np.argmax(valid_power)

    f_hr = valid_frequencies[peak_index]
    hr_bpm = f_hr * 60

    print("\n========== FFT HEART RATE ==========")
    print("Peak frequency:", f_hr, "Hz")
    print("Estimated heart rate:", hr_bpm, "BPM")
    print("====================================")

    # Plot power spectrum in Hz
    plt.figure(figsize=(12, 5))

    plt.plot(valid_frequencies, valid_power)

    plt.axvline(
        f_hr,
        linestyle="--",
        label=f"Peak = {f_hr:.3f} Hz"
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("rPPG Power Spectrum")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Plot power spectrum in BPM
    frequencies_bpm = valid_frequencies * 60

    plt.figure(figsize=(12, 5))

    plt.plot(frequencies_bpm, valid_power)

    plt.axvline(
        hr_bpm,
        linestyle="--",
        label=f"Estimated HR = {hr_bpm:.2f} BPM"
    )

    plt.xlabel("Heart Rate (BPM)")
    plt.ylabel("Power")
    plt.title("rPPG Power Spectrum in BPM")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

    return hr_bpm


# Function call
# heart_rate = estimate_heart_rate("filtered_rppg.csv", FPS=28.671786)

# print("Final Estimated Heart Rate:", heart_rate, "BPM")
