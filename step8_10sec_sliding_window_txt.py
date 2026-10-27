import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def read_ground_truth_txt(gt_file):
    """
    Read Dataset 2 ground_truth.txt.

    Expected format:
      Line 1: PPG signal samples
      Line 2: Heart-rate (HR) samples in BPM
      Line 3: timestamps in seconds (scientific notation supported)

    Each line may be comma-separated or whitespace-separated.
    """
    gt_path = Path(gt_file).resolve()

    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    lines = [
        line.strip()
        for line in gt_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if len(lines) < 3:
        raise ValueError(
            "ground_truth.txt must contain three non-empty lines: PPG, HR, timestamps."
        )

    def parse_line(line, name):
        cleaned = (
            line.replace(",", " ")
                .replace("[", " ")
                .replace("]", " ")
                .replace("(", " ")
                .replace(")", " ")
        )
        values = np.fromstring(cleaned, sep=" ", dtype=np.float64)
        if values.size == 0:
            raise ValueError(f"No numeric values found on the {name} line.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinite values.")
        return values

    ppg = parse_line(lines[0], "PPG")
    hr = parse_line(lines[1], "HR")
    time_sec = parse_line(lines[2], "timestamps")

    if not (len(ppg) == len(hr) == len(time_sec)):
        raise ValueError(
            "PPG, HR, and timestamp lines must have equal sample counts. "
            f"Got PPG={len(ppg)}, HR={len(hr)}, timestamps={len(time_sec)}."
        )
    if len(time_sec) < 2:
        raise ValueError("At least two timestamp samples are required.")
    if np.any(np.diff(time_sec) < 0):
        raise ValueError("Ground-truth timestamps must be in non-decreasing order.")

    return pd.DataFrame({
        "Time_sec": time_sec,
        "HR": hr,
        "PPG": ppg,
    })


def evaluate_sliding_hr(
    rppg_file,
    gt_file,
    # FPS=28.671786,
    FPS,
    WINDOW_SECONDS=15,
    HOP_SECONDS=5,
    LOW_FREQ=0.7,
    HIGH_FREQ=4.0,
):
    """
    Evaluate rPPG heart rate using overlapping/sliding windows against
    Dataset 2's three-line ground_truth.txt file.

    ground_truth.txt:
      Line 1 = PPG
      Line 2 = HR in BPM
      Line 3 = timestamps in seconds
    """
    rppg_path = Path(rppg_file).resolve()
    if not rppg_path.is_file():
        raise FileNotFoundError(f"rPPG CSV file not found: {rppg_path}")

    # 1. Load camera rPPG and TXT ground truth.
    rppg_df = pd.read_csv(rppg_path)
    if "Filtered_rPPG" not in rppg_df.columns:
        raise ValueError("rPPG CSV must contain a 'Filtered_rPPG' column.")

    gt = read_ground_truth_txt(gt_file)
    signal = rppg_df["Filtered_rPPG"].to_numpy(dtype=np.float64)
    gt_time = gt["Time_sec"].to_numpy(dtype=np.float64)
    gt_hr = gt["HR"].to_numpy(dtype=np.float64)

    # 2. Validate parameters and data.
    if FPS <= 0 or WINDOW_SECONDS <= 0 or HOP_SECONDS <= 0:
        raise ValueError("FPS, window duration, and hop duration must be positive.")
    if not (0 < LOW_FREQ < HIGH_FREQ < FPS / 2):
        raise ValueError("Require 0 < LOW_FREQ < HIGH_FREQ < FPS/2.")
    if len(signal) == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("rPPG signal is empty or contains NaN/infinite values.")

    duration = len(signal) / FPS
    gt_start_time = float(gt_time[0])
    gt_duration = float(gt_time[-1] - gt_time[0])
    common_duration = min(duration, gt_duration)

    print("rPPG duration:", duration, "seconds")
    print("Ground-truth duration:", gt_duration, "seconds")
    print("Ground truth:", Path(gt_file).name)
    print("HR window:", WINDOW_SECONDS, "seconds")
    print("Window step:", HOP_SECONDS, "seconds")

    if common_duration < WINDOW_SECONDS:
        raise ValueError("Common recording duration is shorter than one HR window.")

    # 3. Estimate HR in overlapping windows.
    results = []
    last_start = int(np.floor(common_duration - WINDOW_SECONDS + 1e-9))

    for start_sec in range(0, last_start + 1, HOP_SECONDS):
        end_sec = start_sec + WINDOW_SECONDS
        start_idx = int(round(start_sec * FPS))
        end_idx = int(round(end_sec * FPS))
        segment = signal[start_idx:end_idx]

        if len(segment) < 3:
            continue

        segment = segment - np.mean(segment)
        segment = segment * np.hanning(len(segment))

        spectrum = np.fft.rfft(segment)
        frequencies = np.fft.rfftfreq(len(segment), d=1.0 / FPS)
        power = np.abs(spectrum) ** 2

        valid_indices = np.flatnonzero(
            (frequencies >= LOW_FREQ) & (frequencies <= HIGH_FREQ)
        )
        if len(valid_indices) == 0:
            continue

        k = int(valid_indices[np.argmax(power[valid_indices])])
        f_bin = float(frequencies[k])
        f_interp = f_bin

        # Parabolic interpolation on log power.
        if 0 < k < len(power) - 1:
            log_power = np.log(power + 1e-12)
            left = log_power[k - 1]
            center = log_power[k]
            right = log_power[k + 1]
            denominator = left - 2.0 * center + right

            if abs(denominator) > 1e-12:
                delta = 0.5 * (left - right) / denominator
                delta = float(np.clip(delta, -0.5, 0.5))
                f_interp = (k + delta) * FPS / len(segment)

        f_interp = float(np.clip(f_interp, LOW_FREQ, HIGH_FREQ))
        hr_bin = f_bin * 60.0
        hr_interp = f_interp * 60.0

        # Ground-truth timestamps are in seconds, not milliseconds.
        gt_window_start = gt_start_time + start_sec
        gt_window_end = gt_start_time + end_sec
        mask = (gt_time >= gt_window_start) & (gt_time < gt_window_end)
        gt_segment = gt_hr[mask]
        gt_segment = gt_segment[np.isfinite(gt_segment)]

        if len(gt_segment) == 0:
            print(f"Skipping {start_sec}-{end_sec}s: no ground-truth HR samples.")
            continue

        gt_mean = float(np.mean(gt_segment))

        results.append({
            "Start_sec": start_sec,
            "End_sec": end_sec,
            "Center_sec": (start_sec + end_sec) / 2.0,
            "HR_Bin_BPM": hr_bin,
            "HR_Interpolated_BPM": hr_interp,
            "GT_Mean_BPM": gt_mean,
            "Error_Bin_BPM": abs(hr_bin - gt_mean),
            "Error_Interpolated_BPM": abs(hr_interp - gt_mean),
        })

    # 4. Print and save results.
    result_df = pd.DataFrame(results)
    print("\n========== SLIDING WINDOW HR ==========")

    if result_df.empty:
        raise RuntimeError("No valid HR windows were evaluated.")

    print(result_df.round(2).to_string(index=False))

    output_file = Path("sliding_hr_comparison_txt_dataset.csv").resolve()
    result_df.to_csv(output_file, index=False)
    print("\nSaved:", output_file)

    # 5. Calculate metrics.
    errors_bin = (
        result_df["HR_Bin_BPM"] - result_df["GT_Mean_BPM"]
    ).to_numpy()
    errors_interp = (
        result_df["HR_Interpolated_BPM"] - result_df["GT_Mean_BPM"]
    ).to_numpy()

    mae_bin = float(np.mean(np.abs(errors_bin)))
    mae_interp = float(np.mean(np.abs(errors_interp)))
    rmse_bin = float(np.sqrt(np.mean(errors_bin ** 2)))
    rmse_interp = float(np.sqrt(np.mean(errors_interp ** 2)))

    print("\n========== METRICS ==========")
    print("Number of windows:", len(result_df))
    print("Bin-based MAE:", round(mae_bin, 3), "BPM")
    print("Interpolated MAE:", round(mae_interp, 3), "BPM")
    print("Bin-based RMSE:", round(rmse_bin, 3), "BPM")
    print("Interpolated RMSE:", round(rmse_interp, 3), "BPM")
    print("=============================")

    # 6. Plot HR estimates against ground truth.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        result_df["Center_sec"], result_df["GT_Mean_BPM"],
        marker="o", label="Ground-truth mean HR",
    )
    ax.plot(
        result_df["Center_sec"], result_df["HR_Bin_BPM"],
        marker="s", label="FFT bin estimate",
    )
    ax.plot(
        result_df["Center_sec"], result_df["HR_Interpolated_BPM"],
        marker="^", label="Interpolated FFT estimate",
    )
    ax.set_xlabel("Window center time (seconds)")
    ax.set_ylabel("Heart Rate (BPM)")
    ax.set_title("Sliding-Window rPPG HR vs Ground Truth (TXT Dataset)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    plot_file = Path("sliding_hr_comparison_txt_dataset.png").resolve()
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved plot:", plot_file)

    return result_df, mae_bin, mae_interp, rmse_bin, rmse_interp


# if __name__ == "__main__":
#     # Keep this script in the same folder as these files, or change the paths.
#     result_df, mae_bin, mae_interp, rmse_bin, rmse_interp = evaluate_sliding_hr(
#         rppg_file="filtered_rppg.csv",
#         gt_file="ground_truth_1.txt",
#         FPS=28.671786,
#         WINDOW_SECONDS=15,
#         HOP_SECONDS=5,
#         LOW_FREQ=0.7,
#         HIGH_FREQ=4.0,
#     )
