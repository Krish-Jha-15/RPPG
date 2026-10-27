import cv2
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. Check the extracted RGB file
# ------------------------------------------------------------

df = pd.read_csv("average_rgb.csv")

frames = df["Frame"].to_numpy(dtype=int)

print("========== RGB FRAME CHECK ==========")
print("Number of RGB rows:", len(frames))

if len(frames) == 0:
    raise ValueError("average_rgb.csv contains no rows.")

print("First saved frame:", frames[0])
print("Last saved frame:", frames[-1])

frame_diffs = np.diff(frames)

gap_indices = np.where(frame_diffs > 1)[0]

missing_count = (
    frames[-1] - frames[0] + 1 - len(np.unique(frames))
)

print("Missing frame numbers within saved range:", missing_count)
print("Number of gaps:", len(gap_indices))

if len(gap_indices) > 0:
    print("\nFirst 20 gaps:")
    for i in gap_indices[:20]:
        print(
            f"Frame {frames[i]} -> Frame {frames[i + 1]} "
            f"(gap of {frame_diffs[i] - 1} frame(s))"
        )
else:
    print("Frame numbers are continuous.")

print("=====================================")


# ------------------------------------------------------------
# 2. Compare with the original video metadata
# ------------------------------------------------------------

video_path = "vid2.avi"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

video_fps = cap.get(cv2.CAP_PROP_FPS)
video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()

print("\n========== VIDEO CHECK ==========")
print("Video frame count:", video_frames)
print("Video FPS:", video_fps)
print("RGB rows:", len(frames))
print("Difference in counts:", video_frames - len(frames))
print("=================================")