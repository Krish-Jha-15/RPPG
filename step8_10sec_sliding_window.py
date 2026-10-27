import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def evaluate_sliding_hr(
    rppg_file,
    gt_file,
    # FPS=28.671786,
    FPS,
    WINDOW_SECONDS=15,
    HOP_SECONDS=5,
    LOW_FREQ=0.7,
    HIGH_FREQ=4.0
):

    # 1. Load signal and ground truth
    rppg_df = pd.read_csv(rppg_file)
    gt = pd.read_csv(gt_file, header=None)

    gt.columns = ["Timestamp_ms", "HR", "SpO2", "PPG"]
    gt["Time_sec"] = gt["Timestamp_ms"] / 1000.0

    signal = rppg_df["Filtered_rPPG"].to_numpy(
        dtype=np.float64
    )

    # 2. Parameters
    duration = len(signal) / FPS

    print("Signal duration:", duration, "seconds")
    print("HR window:", WINDOW_SECONDS, "seconds")
    print("Window step:", HOP_SECONDS, "seconds")

    if FPS <= 0 or WINDOW_SECONDS <= 0 or HOP_SECONDS <= 0:
        raise ValueError("FPS, window duration, and hop duration must be positive.")

    if not (0 < LOW_FREQ < HIGH_FREQ < FPS / 2):
        raise ValueError("Invalid frequency range for the sampling frequency.")

    if len(signal) == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("Signal is empty or contains NaN/infinite values.")

    # 3. Estimate HR in overlapping windows
    results = []

    last_start = int(duration - WINDOW_SECONDS)

    for start_sec in range(0, last_start + 1, HOP_SECONDS):

        end_sec = start_sec + WINDOW_SECONDS

        start_idx = round(start_sec * FPS)
        end_idx = round(end_sec * FPS)

        segment = signal[start_idx:end_idx]

        if len(segment) < 3:
            continue

        # Remove DC component
        segment = segment - np.mean(segment)

        # Apply Hann window
        segment = segment * np.hanning(len(segment))

        # FFT and frequency axis
        spectrum = np.fft.rfft(segment)
        frequencies = np.fft.rfftfreq(
            len(segment),
            d=1 / FPS
        )

        power = np.abs(spectrum) ** 2

        # Restrict search to plausible HR frequencies
        valid_indices = np.flatnonzero(
            (frequencies >= LOW_FREQ) &
            (frequencies <= HIGH_FREQ)
        )

        if len(valid_indices) == 0:
            continue

        # Strongest valid FFT bin
        k = valid_indices[np.argmax(power[valid_indices])]

        f_bin = frequencies[k]
        f_interp = f_bin

        # Parabolic interpolation on log power
        if 0 < k < len(power) - 1:

            log_power = np.log(power + 1e-12)

            left = log_power[k - 1]
            center = log_power[k]
            right = log_power[k + 1]

            denominator = left - 2 * center + right

            if abs(denominator) > 1e-12:

                delta = 0.5 * (left - right) / denominator
                delta = np.clip(delta, -0.5, 0.5)

                f_interp = (k + delta) * FPS / len(segment)

        # Keep interpolated estimate in allowed frequency range
        f_interp = np.clip(f_interp, LOW_FREQ, HIGH_FREQ)

        hr_bin = f_bin * 60
        hr_interp = f_interp * 60

        # Ground truth for the same time interval
        mask = (
            (gt["Time_sec"] >= start_sec) &
            (gt["Time_sec"] < end_sec)
        )

        gt_segment = gt.loc[mask, "HR"].dropna()

        if gt_segment.empty:
            continue

        gt_mean = gt_segment.mean()

        # Store results
        results.append({
            "Start_sec": start_sec,
            "End_sec": end_sec,
            "Center_sec": (start_sec + end_sec) / 2,
            "HR_Bin_BPM": hr_bin,
            "HR_Interpolated_BPM": hr_interp,
            "GT_Mean_BPM": gt_mean,
            "Error_Bin_BPM": abs(hr_bin - gt_mean),
            "Error_Interpolated_BPM": abs(hr_interp - gt_mean)
        })

    # 4. Print and save results
    result_df = pd.DataFrame(results)

    print("\n========== SLIDING WINDOW HR ==========")

    if result_df.empty:
        raise RuntimeError("No valid HR windows were evaluated.")

    print(result_df.round(2).to_string(index=False))

    output_file = "sliding_hr_comparison.csv"
    result_df.to_csv(output_file, index=False)

    print("\nSaved:", output_file)

    # 5. Calculate metrics
    errors_bin = (
        result_df["HR_Bin_BPM"] -
        result_df["GT_Mean_BPM"]
    ).to_numpy()

    errors_interp = (
        result_df["HR_Interpolated_BPM"] -
        result_df["GT_Mean_BPM"]
    ).to_numpy()

    mae_bin = np.mean(np.abs(errors_bin))
    mae_interp = np.mean(np.abs(errors_interp))

    rmse_bin = np.sqrt(np.mean(errors_bin ** 2))
    rmse_interp = np.sqrt(np.mean(errors_interp ** 2))

    print("\n========== METRICS ==========")
    print("Number of windows:", len(result_df))
    print("Bin-based MAE:", mae_bin, "BPM")
    print("Interpolated MAE:", mae_interp, "BPM")
    print("Bin-based RMSE:", rmse_bin, "BPM")
    print("Interpolated RMSE:", rmse_interp, "BPM")
    print("=============================")

    # 6. Plot HR estimates against ground truth
    plt.figure(figsize=(12, 5))

    plt.plot(
        result_df["Center_sec"],
        result_df["GT_Mean_BPM"],
        marker="o",
        label="UBFC ground-truth mean"
    )

    plt.plot(
        result_df["Center_sec"],
        result_df["HR_Bin_BPM"],
        marker="s",
        label="FFT bin estimate"
    )

    plt.plot(
        result_df["Center_sec"],
        result_df["HR_Interpolated_BPM"],
        marker="^",
        label="Interpolated FFT estimate"
    )

    plt.xlabel("Window center time (seconds)")
    plt.ylabel("Heart Rate (BPM)")
    plt.title("Sliding-Window rPPG HR vs UBFC Ground Truth")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return result_df, mae_bin, mae_interp, rmse_bin, rmse_interp


# Function call
# result_df, mae_bin, mae_interp, rmse_bin, rmse_interp = (
#     evaluate_sliding_hr(
#         rppg_file="filtered_rppg.csv",
#         gt_file="gtdump_5.xmp"
#     )
# )
