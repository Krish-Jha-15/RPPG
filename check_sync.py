import cv2
import pandas as pd

# -------------------------------------------------
# 1. Inspect the video
# -------------------------------------------------

cap = cv2.VideoCapture("vid2.avi")

if not cap.isOpened():
    raise RuntimeError("Could not open vid.avi")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()

video_duration = frame_count / fps

# -------------------------------------------------
# 2. Inspect the matching ground-truth file
# -------------------------------------------------

gt = pd.read_csv("gtdump_2.xmp", header=None)

gt.columns = ["Timestamp_ms", "HR", "SpO2", "PPG"]

gt_start = gt["Timestamp_ms"].iloc[0] / 1000
gt_end = gt["Timestamp_ms"].iloc[-1] / 1000
gt_duration = gt_end - gt_start

# -------------------------------------------------
# 3. Print comparison
# -------------------------------------------------

print("========== SYNCHRONIZATION CHECK ==========")

print("\nVIDEO")
print("FPS:", fps)
print("Frames:", frame_count)
print("Duration:", round(video_duration, 3), "seconds")

print("\nGROUND TRUTH")
print("Rows:", len(gt))
print("Start:", gt_start, "seconds")
print("End:", gt_end, "seconds")
print("Duration:", round(gt_duration, 3), "seconds")

print("\nDuration difference:",
      round(abs(video_duration - gt_duration), 3),
      "seconds")

print("\nGround-truth HR mean:",
      round(gt["HR"].mean(), 2), "BPM")

print("===========================================")