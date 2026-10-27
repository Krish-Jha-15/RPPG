from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = BASE_DIR / "vid_4.avi"
MODEL_PATH = BASE_DIR / "blaze_face_short_range.tflite"

OUTPUT_DIR = BASE_DIR / "vid4_roi_diagnostic_frames"

TARGET_SECONDS = list(range(0, 71, 5))

MIN_DETECTION_CONFIDENCE = 0.5


# ============================================================
# 2. VERIFY FILES
# ============================================================

if not VIDEO_PATH.is_file():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH.resolve()}"
    )

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"MediaPipe model not found: {MODEL_PATH.resolve()}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def clip_box(x1, y1, x2, y2, width, height):
    """Clip a box to valid image boundaries."""

    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))

    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))

    return x1, y1, x2, y2


def draw_box(image, box, color, label, thickness=2):
    """Draw a rectangle and label on an image."""

    x1, y1, x2, y2 = box

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )

    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# 4. OPEN VIDEO AND INITIALIZE MEDIAPIPE
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH.resolve()}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

if fps <= 0 or frame_count <= 0:
    cap.release()
    raise RuntimeError("Invalid video FPS or frame count.")

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE
)

records = []

print("Video:", VIDEO_PATH.resolve())
print("FPS:", fps)
print("Frames:", frame_count)
print("Output folder:", OUTPUT_DIR.resolve())


# ============================================================
# 5. PROCESS SELECTED FRAMES
# ============================================================

target_frames = {
    round(seconds * fps): seconds
    for seconds in TARGET_SECONDS
}

max_target_frame = max(target_frames)

try:
    with FaceDetector.create_from_options(options) as detector:

        frame_index = 0

        while frame_index <= max_target_frame:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_index not in target_frames:
                frame_index += 1
                continue

            target_sec = target_frames[frame_index]

            height, width = frame.shape[:2]

            # MediaPipe expects RGB.
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            results = detector.detect(mp_image)

            row = {
                "Time_sec": target_sec,
                "Frame": frame_index + 1,
                "Face_detected": False
            }

            # ------------------------------------------------
            # FACE DETECTION
            # ------------------------------------------------

            if not results.detections:

                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

                print(
                    f"No face detected at {target_sec} seconds."
                )

            else:

                # Match the existing implementation:
                # select the first detected face.
                detection = results.detections[0]

                bbox = detection.bounding_box

                x = bbox.origin_x
                y = bbox.origin_y
                bw = bbox.width
                bh = bbox.height

                # Draw the original detector bounding box.
                original_box = clip_box(
                    x, y,
                    x + bw, y + bh,
                    width, height
                )

                draw_box(
                    frame,
                    original_box,
                    (0, 255, 0),
                    "Detected face"
                )

                # ------------------------------------------------
                # EXPANDED FACE REFERENCE BOX
                # Same proportions as existing ROI code.
                # ------------------------------------------------

                face_x1 = int(x - 0.15 * bw)
                face_x2 = int(x + 1.15 * bw)

                face_y1 = int(y - 0.35 * bh)
                face_y2 = int(y + 1.10 * bh)

                face_x1, face_y1, face_x2, face_y2 = (
                    clip_box(
                        face_x1,
                        face_y1,
                        face_x2,
                        face_y2,
                        width,
                        height
                    )
                )

                face_width = face_x2 - face_x1
                face_height = face_y2 - face_y1

                if face_width <= 0 or face_height <= 0:
                    print(
                        f"Invalid face box at {target_sec} seconds."
                    )
                    frame_index += 1
                    continue

                # ------------------------------------------------
                # FOREHEAD ROI
                # ------------------------------------------------

                forehead_box = (
                    face_x1 + int(0.20 * face_width),
                    face_y1 + int(0.08 * face_height),
                    face_x1 + int(0.80 * face_width),
                    face_y1 + int(0.30 * face_height)
                )

                forehead_box = clip_box(
                    *forehead_box,
                    width,
                    height
                )

                # ------------------------------------------------
                # LEFT CHEEK ROI
                # ------------------------------------------------

                left_cheek_box = (
                    face_x1 + int(0.12 * face_width),
                    face_y1 + int(0.48 * face_height),
                    face_x1 + int(0.38 * face_width),
                    face_y1 + int(0.70 * face_height)
                )

                left_cheek_box = clip_box(
                    *left_cheek_box,
                    width,
                    height
                )

                # ------------------------------------------------
                # RIGHT CHEEK ROI
                # ------------------------------------------------

                right_cheek_box = (
                    face_x1 + int(0.62 * face_width),
                    face_y1 + int(0.48 * face_height),
                    face_x1 + int(0.88 * face_width),
                    face_y1 + int(0.70 * face_height)
                )

                right_cheek_box = clip_box(
                    *right_cheek_box,
                    width,
                    height
                )

                # ------------------------------------------------
                # DRAW ROI BOXES
                # OpenCV drawing colors are BGR.
                # ------------------------------------------------

                draw_box(
                    frame,
                    forehead_box,
                    (255, 0, 0),
                    "Forehead"
                )

                draw_box(
                    frame,
                    left_cheek_box,
                    (0, 255, 255),
                    "Left cheek"
                )

                draw_box(
                    frame,
                    right_cheek_box,
                    (0, 255, 255),
                    "Right cheek"
                )

                # ------------------------------------------------
                # RECORD ROI GEOMETRY
                # ------------------------------------------------

                def box_dimensions(box):
                    x1, y1, x2, y2 = box
                    return x2 - x1, y2 - y1

                fw, fh = box_dimensions(forehead_box)
                lw, lh = box_dimensions(left_cheek_box)
                rw, rh = box_dimensions(right_cheek_box)

                row.update({
                    "Face_detected": True,

                    "Face_x1": face_x1,
                    "Face_y1": face_y1,
                    "Face_x2": face_x2,
                    "Face_y2": face_y2,

                    "Forehead_width": fw,
                    "Forehead_height": fh,

                    "Left_cheek_width": lw,
                    "Left_cheek_height": lh,

                    "Right_cheek_width": rw,
                    "Right_cheek_height": rh
                })

            # Add timestamp to the image.
            cv2.putText(
                frame,
                f"Time: {target_sec}s",
                (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            output_path = (
                OUTPUT_DIR /
                f"frame_{frame_index + 1:05d}_{target_sec:02d}s.jpg"
            )

            saved = cv2.imwrite(
                str(output_path),
                frame
            )

            if not saved:
                raise RuntimeError(
                    f"Failed to save image: {output_path}"
                )

            records.append(row)

            print(
                f"Saved {output_path.name}"
            )

            frame_index += 1

finally:
    cap.release()


# ============================================================
# 6. SAVE ROI DIAGNOSTIC TABLE
# ============================================================

diagnostic_df = pd.DataFrame(records)

csv_path = OUTPUT_DIR / "roi_diagnostic.csv"

diagnostic_df.to_csv(
    csv_path,
    index=False
)

print("\n========== ROI DIAGNOSTIC COMPLETE ==========")
print("Selected frames:", len(diagnostic_df))
print("Detected faces:",
      int(diagnostic_df["Face_detected"].sum()))
print("Saved table:", csv_path.resolve())
print("Saved images:", OUTPUT_DIR.resolve())
print("=============================================")
