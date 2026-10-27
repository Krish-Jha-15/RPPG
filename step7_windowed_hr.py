
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def evaluate_windowed_hr(
    input_csv,
    ground_truth_file,
    # fps=28.671786,
    fps,
    window_seconds=10,
    low_freq=0.7,
    high_freq=4.0
):

    input_path = Path(input_csv).resolve()
    gt_path = Path(ground_truth_file).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Filtered rPPG file not found: {input_path}"
        )

    if not gt_path.is_file():
        raise FileNotFoundError(
            f"Ground-truth file not found: {gt_path}"
        )

    df = pd.read_csv(input_path)

    if "Filtered_rPPG" not in df.columns:
        raise ValueError(
            "Input CSV must contain a 'Filtered_rPPG' column."
        )

    signal = df["Filtered_rPPG"].to_numpy(dtype=np.float64)

    if len(signal) == 0:
        raise ValueError("Filtered rPPG signal is empty.")

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "Filtered rPPG contains NaN or infinite values."
        )

    if fps <= 0 or window_seconds <= 0:
        raise ValueError("FPS and window duration must be positive.")

    signal_duration = len(signal) / fps

    gt = pd.read_csv(gt_path, header=None)

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

    common_duration = min(
        signal_duration,
        gt_time[-1]
    )

    number_of_windows = int(
        np.floor(
            (common_duration + 1e-9) / window_seconds
        )
    )

    print("Sampling frequency:", fps)
    print("Total rPPG samples:", len(signal))
    print("Signal duration:", signal_duration, "seconds")
    print("Ground truth:", gt_path.name)
    print("HR window:", window_seconds, "seconds")
    print("Complete windows available:", number_of_windows)

    if number_of_windows == 0:
        raise ValueError(
            "Recording is shorter than one complete HR window."
        )

    results = []

    for window_index in range(number_of_windows):

        start_sec = window_index * window_seconds
        end_sec = start_sec + window_seconds

        start_idx = round(start_sec * fps)
        end_idx = round(end_sec * fps)

        segment = signal[start_idx:end_idx]

        # Protect against an unexpected short segment.
        if len(segment) < 2:
            continue

        # Remove DC component.
        segment = segment - np.mean(segment)

        # Reduce spectral leakage.
        window = np.hanning(len(segment))
        windowed_segment = segment * window

        # Calculate FFT.
        fft_result = np.fft.rfft(windowed_segment)

        frequencies = np.fft.rfftfreq(
            len(windowed_segment),
            d=1.0 / fps
        )

        power = np.abs(fft_result) ** 2

        # Search only within the allowed HR frequency range.
        valid_mask = (
            (frequencies >= low_freq) &
            (frequencies <= high_freq)
        )

        valid_frequencies = frequencies[valid_mask]
        valid_power = power[valid_mask]

        if len(valid_power) == 0:
            continue

        peak_index = int(np.argmax(valid_power))

        estimated_hr = (
            valid_frequencies[peak_index] * 60.0
        )

        gt_mask = (
            (gt_time >= start_sec) &
            (gt_time < end_sec)
        )

        gt_segment = gt_hr[gt_mask]

        if len(gt_segment) == 0:
            print(
                f"Skipping {start_sec}-{end_sec}s: "
                "no ground-truth samples."
            )
            continue

        gt_mean_hr = float(np.mean(gt_segment))
        gt_median_hr = float(np.median(gt_segment))

        absolute_error = abs(
            estimated_hr - gt_mean_hr
        )

        results.append({
            "Start_sec": start_sec,
            "End_sec": end_sec,
            "Samples": len(segment),
            "Estimated_HR_BPM": estimated_hr,
            "GT_Mean_HR_BPM": gt_mean_hr,
            "GT_Median_HR_BPM": gt_median_hr,
            "Absolute_Error_BPM": absolute_error
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise RuntimeError(
            "No complete windows could be evaluated."
        )

    print("\n========== COMPLETE WINDOW RESULTS ==========")
    print(results_df.round(2).to_string(index=False))

    # Save inside the current results directory.
    output_csv = Path("windowed_hr_comparison.csv").resolve()

    results_df.to_csv(
        output_csv,
        index=False
    )

    print("\nSaved:", output_csv)


    errors = (
        results_df["Estimated_HR_BPM"].to_numpy()
        - results_df["GT_Mean_HR_BPM"].to_numpy()
    )

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    errors_abs = np.abs(errors)
    within_5_count = int(np.sum(errors_abs <= 5.0))
    within_5_percent = 100.0 * within_5_count / len(errors)

    print("\n========== EVALUATION ==========")
    print("Number of complete windows:", len(results_df))
    print("MAE:", round(mae, 3), "BPM")
    print("RMSE:", round(rmse, 3), "BPM")
    print(
        "Windows within +/-5 BPM:",
        f"{within_5_count}/{len(errors)}",
        f"({within_5_percent:.1f}%)"
    )
    print("================================")

    centers = (
        results_df["Start_sec"] +
        results_df["End_sec"]
    ) / 2.0

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        centers,
        results_df["Estimated_HR_BPM"],
        marker="o",
        label="rPPG estimated HR"
    )

    ax.plot(
        centers,
        results_df["GT_Mean_HR_BPM"],
        marker="s",
        label="Ground-truth mean HR"
    )

    ax.set_xlabel("Window center (seconds)")
    ax.set_ylabel("Heart Rate (BPM)")
    ax.set_title(
        f"Non-overlapping {window_seconds}-second HR evaluation"
    )

    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    plot_path = Path(
        "windowed_hr_comparison.png"
    ).resolve()

    fig.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved plot:", plot_path)

    return results_df

# Function call
# results_df, mae, rmse = evaluate_windowed_hr(
#     input_csv="filtered_rppg.csv",
#     ground_truth_file="gtdump_5.xmp"
# )
