
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, detrend


def compare_three_hr_signals(
    rppg_file,
    gt_file,
    # FPS=28.671786,
    FPS,
    WINDOW_SECONDS=15,
    HOP_SECONDS=5,
    LOW_FREQ=0.7,
    HIGH_FREQ=4.0
):

    # ==========================================
    # 1. Load camera rPPG
    # ==========================================

    rppg_df = pd.read_csv(rppg_file)

    rppg = rppg_df["Filtered_rPPG"].to_numpy(dtype=float)

    # ==========================================
    # 2. Load Dataset 2 ground_truth.txt
    # ==========================================

    with open(gt_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) != 3:
        raise ValueError(
            "Expected exactly 3 non-empty lines: PPG, HR, Timestamp."
        )

    # Line 1: PPG signal
    gt_ppg = np.fromstring(lines[0], sep=" ")

    # Line 2: Heart rate
    gt_hr = np.fromstring(lines[1], sep=" ")

    # Line 3: Timestamp (already in seconds)
    gt_time = np.fromstring(lines[2], sep=" ")

    if not (len(gt_ppg) == len(gt_hr) == len(gt_time)):
        raise ValueError(
            "PPG, HR and Timestamp must have equal lengths."
        )

    if len(rppg) == 0 or len(gt_time) < 2:
        raise ValueError("Insufficient signal data.")

    if not np.all(np.isfinite(rppg)):
        raise ValueError("Camera rPPG contains invalid values.")

    if not np.all(np.isfinite(gt_ppg)):
        raise ValueError("Reference PPG contains invalid values.")

    if not np.all(np.isfinite(gt_time)):
        raise ValueError("Invalid ground-truth timestamps.")

   # Remove consecutive duplicate timestamps
    valid_mask = np.concatenate((
        [True],
        np.diff(gt_time) != 0
    ))

    gt_time = gt_time[valid_mask]
    gt_hr = gt_hr[valid_mask]
    gt_ppg = gt_ppg[valid_mask]

    print("Samples after removing duplicates:", len(gt_time))

    # Validate timestamps again
    if not np.all(np.diff(gt_time) > 0):
        raise ValueError(
            "Ground-truth timestamps must be strictly increasing."
        )
    
    if not (0 < LOW_FREQ < HIGH_FREQ < FPS / 2):
        raise ValueError("Invalid camera frequency range.")

    # ==========================================
    # 3. Reference PPG sampling frequency
    # ==========================================

    gt_fs = 1.0 / np.median(np.diff(gt_time))

    if not (0 < LOW_FREQ < HIGH_FREQ < gt_fs / 2):
        raise ValueError("Reference PPG sampling rate is too low.")

    print("Camera FPS:", FPS)
    print("Reference PPG sampling rate:", gt_fs)
    print("Ground truth samples:", len(gt_time))

    # ==========================================
    # 4. Resample and filter reference PPG
    # ==========================================

    uniform_time = np.arange(
        gt_time[0],
        gt_time[-1],
        1.0 / gt_fs
    )

    uniform_ppg = np.interp(
        uniform_time,
        gt_time,
        gt_ppg
    )

    uniform_ppg = detrend(uniform_ppg, type="linear")

    sos = butter(
        4,
        [LOW_FREQ, HIGH_FREQ],
        btype="bandpass",
        fs=gt_fs,
        output="sos"
    )

    filtered_reference_ppg = sosfiltfilt(
        sos,
        uniform_ppg
    )

    # ==========================================
    # 5. Estimate HR using FFT
    # ==========================================

    def estimate_hr(segment, fs):

        segment = np.asarray(segment, dtype=float)

        if len(segment) < 3 or not np.all(np.isfinite(segment)):
            return np.nan

        segment = segment - np.mean(segment)

        segment = segment * np.hanning(len(segment))

        spectrum = np.fft.rfft(segment)

        frequencies = np.fft.rfftfreq(
            len(segment),
            d=1.0 / fs
        )

        power = np.abs(spectrum) ** 2

        valid_indices = np.flatnonzero(
            (frequencies >= LOW_FREQ) &
            (frequencies <= HIGH_FREQ)
        )

        if len(valid_indices) == 0:
            return np.nan

        k = valid_indices[np.argmax(power[valid_indices])]

        frequency = frequencies[k]

        # Parabolic interpolation
        if 0 < k < len(power) - 1:

            log_power = np.log(power + 1e-12)

            left = log_power[k - 1]
            center = log_power[k]
            right = log_power[k + 1]

            denominator = left - 2 * center + right

            if abs(denominator) > 1e-12:

                offset = 0.5 * (left - right) / denominator

                offset = np.clip(offset, -0.5, 0.5)

                frequency = (k + offset) * fs / len(segment)

        frequency = np.clip(
            frequency,
            LOW_FREQ,
            HIGH_FREQ
        )

        return frequency * 60.0

    # ==========================================
    # 6. Compare overlapping windows
    # ==========================================

    if WINDOW_SECONDS <= 0 or HOP_SECONDS <= 0:
        raise ValueError("Window and hop must be positive.")

    video_duration = len(rppg) / FPS

    duration = min(
        video_duration,
        gt_time[-1] - gt_time[0]
    )

    print("Common duration:", duration, "seconds")
    print("Window duration:", WINDOW_SECONDS)
    print("Hop duration:", HOP_SECONDS)

    results = []

    for start_sec in np.arange(
        0,
        duration - WINDOW_SECONDS + 1e-9,
        HOP_SECONDS
    ):

        end_sec = start_sec + WINDOW_SECONDS

        # Camera rPPG segment
        r_start = round(start_sec * FPS)
        r_end = round(end_sec * FPS)

        r_segment = rppg[r_start:r_end]

        # Reference PPG segment
        ppg_mask = (
            (uniform_time >= gt_time[0] + start_sec) &
            (uniform_time < gt_time[0] + end_sec)
        )

        ppg_segment = filtered_reference_ppg[ppg_mask]

        # Supplied ground-truth HR
        hr_mask = (
            (gt_time >= gt_time[0] + start_sec) &
            (gt_time < gt_time[0] + end_sec)
        )

        reference_hr_values = gt_hr[hr_mask]

        reference_hr_values = reference_hr_values[
            np.isfinite(reference_hr_values)
        ]

        if len(reference_hr_values) == 0:
            continue

        camera_hr = estimate_hr(r_segment, FPS)

        waveform_hr = estimate_hr(ppg_segment, gt_fs)

        supplied_hr_mean = np.mean(reference_hr_values)

        results.append({

            "Start_sec": start_sec,

            "End_sec": end_sec,

            "Center_sec": (start_sec + end_sec) / 2,

            "Camera_rPPG_HR": camera_hr,

            "Reference_PPG_HR": waveform_hr,

            "Supplied_HR_Mean": supplied_hr_mean,

            "Camera_vs_PPG_Error": abs(
                camera_hr - waveform_hr
            ),

            "Camera_vs_HRColumn_Error": abs(
                camera_hr - supplied_hr_mean
            ),

            "PPG_vs_HRColumn_Error": abs(
                waveform_hr - supplied_hr_mean
            )

        })

    # ==========================================
    # 7. Save comparison
    # ==========================================

    result_df = pd.DataFrame(results)

    if result_df.empty:
        raise RuntimeError("No valid HR windows evaluated.")

    print("\n========== THREE-SIGNAL COMPARISON ==========")

    print(result_df.round(2).to_string(index=False))

    output_file = "three_signal_comparison_dataset2.csv"

    result_df.to_csv(output_file, index=False)

    print("\nSaved:", output_file)

    # ==========================================
    # 8. Summary metrics
    # ==========================================

    print("\n========== SUMMARY ==========")

    error_columns = [
        "Camera_vs_PPG_Error",
        "Camera_vs_HRColumn_Error",
        "PPG_vs_HRColumn_Error"
    ]

    for column in error_columns:

        errors = result_df[column].dropna()

        if len(errors) == 0:
            print(column, "MAE: unavailable")

        else:
            print(
                column,
                "MAE:",
                round(errors.mean(), 3),
                "BPM"
            )

    print("==============================")

    # ==========================================
    # 9. Plot all three HR estimates
    # ==========================================

    plt.figure(figsize=(13, 6))

    plt.plot(
        result_df["Center_sec"],
        result_df["Camera_rPPG_HR"],
        marker="o",
        label="Camera-derived rPPG"
    )

    plt.plot(
        result_df["Center_sec"],
        result_df["Reference_PPG_HR"],
        marker="s",
        label="Reference PPG spectrum"
    )

    plt.plot(
        result_df["Center_sec"],
        result_df["Supplied_HR_Mean"],
        marker="^",
        label="Supplied ground-truth HR mean"
    )

    plt.xlabel("Window center time (seconds)")
    plt.ylabel("Heart Rate (BPM)")

    plt.title("Three-Signal HR Comparison - Dataset 2")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return result_df


# ==========================================
# FUNCTION CALL
# ==========================================

# result_df = compare_three_hr_signals(
#     rppg_file="filtered_rppg.csv",
#     gt_file="ground_truth_1.txt",
#     FPS
# )
