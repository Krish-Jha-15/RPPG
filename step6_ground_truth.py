import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compare_heart_rate(input_file, our_hr):

    # Read ground-truth data
    gt = pd.read_csv(input_file, header=None)

    gt.columns = [
        "Timestamp_ms",
        "HR",
        "SpO2",
        "PPG"
    ]

    print("========== GROUND TRUTH INFORMATION ==========")

    print("\nFirst 10 rows:")
    print(gt.head(10))

    print("\nShape:")
    print(gt.shape)

    print("\nLast row:")
    print(gt.tail(1))

    print("==============================================")

    # Convert milliseconds to seconds
    gt["Time_sec"] = gt["Timestamp_ms"] / 1000.0

    # Ground-truth statistics
    print("\n========== GROUND TRUTH STATISTICS ==========")

    print("Start time:", gt["Time_sec"].iloc[0], "seconds")
    print("End time:", gt["Time_sec"].iloc[-1], "seconds")

    duration = (
        gt["Time_sec"].iloc[-1] -
        gt["Time_sec"].iloc[0]
    )

    print("Duration:", duration, "seconds")

    print("\nHR mean:", gt["HR"].mean(), "BPM")
    print("HR median:", gt["HR"].median(), "BPM")
    print("HR minimum:", gt["HR"].min(), "BPM")
    print("HR maximum:", gt["HR"].max(), "BPM")

    print("=============================================")

    # Plot ground-truth heart rate
    plt.figure(figsize=(12, 5))

    plt.plot(gt["Time_sec"], gt["HR"])

    plt.xlabel("Time (seconds)")
    plt.ylabel("Heart Rate (BPM)")
    plt.title("UBFC Ground-Truth Heart Rate")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Compare estimated HR with ground-truth mean
    ground_truth_hr = gt["HR"].mean()

    absolute_error = abs(our_hr - ground_truth_hr)

    print("\n========== HR COMPARISON ==========")

    print("Our rPPG HR:", our_hr, "BPM")
    print("Ground-truth mean HR:", ground_truth_hr, "BPM")
    print("Absolute error:", absolute_error, "BPM")

    print("===================================")

    return ground_truth_hr, absolute_error


# Function call
# our_hr = 70.15934584013051

# ground_truth_hr, absolute_error = compare_heart_rate(
#     "gtdump_5.xmp",
#     70.15934584013051
# )
