import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def read_ground_truth_txt(ground_truth_file):
    """
    Read Dataset 2 ground_truth.txt.

    Expected format:
      Line 1: PPG samples
      Line 2: HR samples (BPM)
      Line 3: timestamps in seconds (scientific notation is supported)

    Values may be comma-separated or whitespace-separated.
    """
    gt_path = Path(ground_truth_file).resolve()

    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    lines = [
        line.strip()
        for line in gt_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if len(lines) < 3:
        raise ValueError(
            "ground_truth.txt must contain at least 3 non-empty lines: "
            "PPG, HR, and timestamps."
        )

    def parse_numeric_line(line, line_name):
        # Supports comma-separated or whitespace-separated values, including
        # scientific notation such as 1.234e-02.
        cleaned = (
            line.replace(",", " ")
                .replace("[", " ")
                .replace("]", " ")
                .replace("(", " ")
                .replace(")", " ")
        )
        try:
            values = np.fromstring(cleaned, sep=" ", dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"Could not parse {line_name} in {gt_path}") from exc

        if values.size == 0:
            raise ValueError(f"No numeric values found on the {line_name} line.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{line_name} contains NaN or infinite values.")
        return values

    gt_ppg = parse_numeric_line(lines[0], "PPG")
    gt_hr = parse_numeric_line(lines[1], "HR")
    gt_time = parse_numeric_line(lines[2], "timestamps")

    if not (len(gt_ppg) == len(gt_hr) == len(gt_time)):
        raise ValueError(
            "The PPG, HR, and timestamp lines must contain the same number "
            f"of values. Got PPG={len(gt_ppg)}, HR={len(gt_hr)}, "
            f"timestamps={len(gt_time)}."
        )

    if len(gt_time) < 2:
        raise ValueError("At least two timestamp samples are required.")
    if np.any(np.diff(gt_time) < 0):
        raise ValueError("Timestamps must be in non-decreasing order.")

    return pd.DataFrame({
        "Time_sec": gt_time,
        "HR": gt_hr,
        "PPG": gt_ppg,
    })


def evaluate_windowed_hr(
    input_csv,
    ground_truth_file,
    # fps=28.671786,
    fps,
    window_seconds=10,
    low_freq=0.7,
    high_freq=4.0,
):
    """
    Evaluate estimated rPPG heart rate against Dataset 2 ground_truth.txt
    using complete, non-overlapping windows.

    input_csv:
        CSV containing a 'Filtered_rPPG' column.
    ground_truth_file:
        Dataset 2 ground_truth.txt:
          line 1 = PPG, line 2 = HR, line 3 = timestamps in seconds.
    fps:
        Camera rPPG sampling frequency.
    window_seconds:
        Duration of each non-overlapping evaluation window.
    low_freq/high_freq:
        Allowed HR frequency range in Hz.
    """
    input_path = Path(input_csv).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Filtered rPPG file not found: {input_path}")
    if fps <= 0 or window_seconds <= 0:
        raise ValueError("FPS and window duration must be positive.")
    if not (0 < low_freq < high_freq):
        raise ValueError("Require 0 < low_freq < high_freq.")

    # 1. Read filtered camera rPPG.
    df = pd.read_csv(input_path)
    if "Filtered_rPPG" not in df.columns:
        raise ValueError("Input CSV must contain a 'Filtered_rPPG' column.")

    signal = df["Filtered_rPPG"].to_numpy(dtype=np.float64)
    if len(signal) == 0:
        raise ValueError("Filtered rPPG signal is empty.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Filtered rPPG contains NaN or infinite values.")

    signal_duration = len(signal) / fps

    # 2. Read Dataset 2 ground truth.
    gt = read_ground_truth_txt(ground_truth_file)
    gt_time = gt["Time_sec"].to_numpy(dtype=np.float64)
    gt_hr = gt["HR"].to_numpy(dtype=np.float64)

    # Use the common time interval. Ground-truth timestamps are already seconds.
    common_duration = min(signal_duration, float(gt_time[-1] - gt_time[0]))
    gt_start_time = float(gt_time[0])

    number_of_windows = int(np.floor((common_duration + 1e-9) / window_seconds))

    print("Sampling frequency:", fps)
    print("Total rPPG samples:", len(signal))
    print("rPPG duration:", round(signal_duration, 3), "seconds")
    print("Ground truth:", Path(ground_truth_file).name)
    print("Ground-truth samples:", len(gt_time))
    print("Ground-truth start time:", gt_start_time, "seconds")
    print("HR window:", window_seconds, "seconds")
    print("Complete windows available:", number_of_windows)

    if number_of_windows == 0:
        raise ValueError("Recording is shorter than one complete HR window.")

    # 3. Estimate HR and compare with ground-truth HR for each window.
    results = []

    for window_index in range(number_of_windows):
        start_sec = window_index * window_seconds
        end_sec = start_sec + window_seconds

        start_idx = int(round(start_sec * fps))
        end_idx = int(round(end_sec * fps))
        segment = signal[start_idx:end_idx]

        if len(segment) < 2:
            continue

        segment = segment - np.mean(segment)
        windowed_segment = segment * np.hanning(len(segment))

        fft_result = np.fft.rfft(windowed_segment)
        frequencies = np.fft.rfftfreq(len(windowed_segment), d=1.0 / fps)
        power = np.abs(fft_result) ** 2

        valid_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
        valid_frequencies = frequencies[valid_mask]
        valid_power = power[valid_mask]

        if len(valid_power) == 0:
            print(f"Skipping {start_sec}-{end_sec}s: no FFT bins in HR range.")
            continue

        estimated_hr = float(valid_frequencies[int(np.argmax(valid_power))] * 60.0)

        # Translate relative window time to the ground-truth timestamp range.
        gt_window_start = gt_start_time + start_sec
        gt_window_end = gt_start_time + end_sec
        gt_mask = (gt_time >= gt_window_start) & (gt_time < gt_window_end)
        gt_segment = gt_hr[gt_mask]

        if len(gt_segment) == 0:
            print(f"Skipping {start_sec}-{end_sec}s: no ground-truth HR samples.")
            continue

        gt_mean_hr = float(np.mean(gt_segment))
        gt_median_hr = float(np.median(gt_segment))

        results.append({
            "Start_sec": start_sec,
            "End_sec": end_sec,
            "Samples": len(segment),
            "Estimated_HR_BPM": estimated_hr,
            "GT_Mean_HR_BPM": gt_mean_hr,
            "GT_Median_HR_BPM": gt_median_hr,
            "Absolute_Error_BPM": abs(estimated_hr - gt_mean_hr),
        })

    # 4. Display and save window results.
    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise RuntimeError("No complete windows could be evaluated.")

    print("\n========== COMPLETE WINDOW RESULTS ==========")
    print(results_df.round(2).to_string(index=False))

    output_csv = Path("windowed_hr_comparison.txt_dataset.csv").resolve()
    results_df.to_csv(output_csv, index=False)
    print("\nSaved:", output_csv)

    # 5. Calculate metrics.
    errors = (
        results_df["Estimated_HR_BPM"].to_numpy()
        - results_df["GT_Mean_HR_BPM"].to_numpy()
    )
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    within_5_count = int(np.sum(np.abs(errors) <= 5.0))
    within_5_percent = 100.0 * within_5_count / len(errors)

    print("\n========== EVALUATION ==========")
    print("Number of evaluated windows:", len(results_df))
    print("MAE:", round(mae, 3), "BPM")
    print("RMSE:", round(rmse, 3), "BPM")
    print(
        "Windows within +/-5 BPM:",
        f"{within_5_count}/{len(errors)} ({within_5_percent:.1f}%)",
    )
    print("================================")

    # 6. Save plot.
    centers = (results_df["Start_sec"] + results_df["End_sec"]) / 2.0

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        centers,
        results_df["Estimated_HR_BPM"],
        marker="o",
        label="rPPG estimated HR",
    )
    ax.plot(
        centers,
        results_df["GT_Mean_HR_BPM"],
        marker="s",
        label="Ground-truth mean HR",
    )
    ax.set_xlabel("Window center (seconds)")
    ax.set_ylabel("Heart Rate (BPM)")
    ax.set_title(f"Non-overlapping {window_seconds}-second HR evaluation")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    plot_path = Path("windowed_hr_comparison.txt_dataset.png").resolve()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved plot:", plot_path)

    return results_df


# if __name__ == "__main__":
#     # Change these two paths to match your files.
#     results_df = evaluate_windowed_hr(
#         input_csv="filtered_rppg.csv",
#         ground_truth_file="data_set/data_txt/ground_truth_1.txt",
#         fps=28.671786,
#         window_seconds=10,
#         low_freq=0.7,
#         high_freq=4.0,
#     )
