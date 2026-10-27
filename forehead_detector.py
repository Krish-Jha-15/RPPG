import cv2
import mediapipe as mp
import os


video_path = "vid.avi"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")


#Model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "blaze_face_short_range.tflite"
)


# -----------------------------
# 3. Initialize MediaPipe
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    min_detection_confidence=0.5
)

face_detector = FaceDetector.create_from_options(options)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect face
    results = face_detector.detect(mp_image)

    if results.detections:

        for detection in results.detections:

            bbox = detection.bounding_box

            # Original face bounding box
            x = bbox.origin_x
            y = bbox.origin_y
            box_width = bbox.width
            box_height = bbox.height

            forehead_x1 = int(x + 0.10 * box_width)
            forehead_x2 = int(x + 0.90 * box_width)

            forehead_y1 = int(y - 0.30 * box_height)
            forehead_y2 = int(y + 0.10 * box_height)


            forehead_x1 = max(0, forehead_x1)
            forehead_y1 = max(0, forehead_y1)

            forehead_x2 = min(w, forehead_x2)
            forehead_y2 = min(h, forehead_y2)

            cv2.rectangle(
                frame,
                (forehead_x1, forehead_y1),
                (forehead_x2, forehead_y2),
                (255, 0, 0),   # Blue in BGR
                2
            )

            cv2.putText(
                frame,
                "Forehead ROI",
                (forehead_x1, max(25, forehead_y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    cv2.imshow(
        "UBFC - Forehead ROI",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
face_detector.close()
cv2.destroyAllWindows()