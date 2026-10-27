import cv2

video_path = "vid_test_1.mp4"

cap = cv2.VideoCapture(video_path)
print(cap)
if not cap.isOpened():
    raise RuntimeError("Could not open the video.")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("Frame count:", frame_count)
print("Resolution:", width, "x", height)

cap.release()