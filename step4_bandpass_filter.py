import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt


def filter_rppg(
    input_file,
    # FPS=28.671786,
    FPS,

    low_cutoff=0.7,
    high_cutoff=4.0
):

    # Read POS signal
    df = pd.read_csv(input_file)
    pos = df["POS"].to_numpy(dtype=np.float64)

    # Validate parameters
    if not (0 < low_cutoff < high_cutoff < FPS / 2):
        raise ValueError(
            "Cutoff frequencies must satisfy "
            "0 < low_cutoff < high_cutoff < Nyquist frequency."
        )

    print("Sampling frequency:", FPS)
    print("Nyquist frequency:", FPS / 2)
    print("Low cutoff:", low_cutoff, "Hz")
    print("High cutoff:", high_cutoff, "Hz")

    # Design Butterworth band-pass filter
    sos = butter(
        N=4,
        Wn=[low_cutoff, high_cutoff],
        btype="bandpass",
        fs=FPS,
        output="sos"
    )

    # Apply zero-phase filtering
    filtered_pos = sosfiltfilt(sos, pos)

    # Create output DataFrame
    filtered_df = pd.DataFrame({
        "Frame": df["Frame"],
        "POS": pos,
        "Filtered_rPPG": filtered_pos
    })

    # Save filtered signal
    output_file = "filtered_rppg.csv"
    filtered_df.to_csv(output_file, index=False)

    print("\nSuccessfully saved:", output_file)

    # Create time axis
    time = np.arange(len(pos)) / FPS

    # Plot raw POS signal
    plt.figure(figsize=(12, 5))
    plt.plot(time, pos)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Raw Windowed POS Signal")
    plt.grid(True)
    plt.tight_layout()

    # Plot filtered rPPG signal
    plt.figure(figsize=(12, 5))
    plt.plot(time, filtered_pos)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Band-pass Filtered rPPG Signal")
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    return filtered_df


# Function call
# filtered_df = filter_rppg("pos_signal_windowed.csv", FPS=28.671786)
