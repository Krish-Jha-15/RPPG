import cv2
import mediapipe as mp
import os

video_path = "vid.avi"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "blaze_face_short_range.tflite"
)

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    min_detection_confidence=0.5
)

face_detector = FaceDetector.create_from_options(options)

def clip_coordinates(x1, y1, x2, y2, w, h):

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))

    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    return x1, y1, x2, y2

while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    # OpenCV BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Face detection
    results = face_detector.detect(mp_image)


    if results.detections:

        for detection in results.detections:

            bbox = detection.bounding_box

            x = bbox.origin_x
            y = bbox.origin_y
            bw = bbox.width
            bh = bbox.height

            face_x1 = int(x - 0.15 * bw)
            face_x2 = int(x + 1.15 * bw)

            face_y1 = int(y - 0.35 * bh)
            face_y2 = int(y + 1.10 * bh)

            face_x1, face_y1, face_x2, face_y2 = clip_coordinates(
                face_x1, face_y1,
                face_x2, face_y2,
                w, h
            )

            face_width = face_x2 - face_x1
            face_height = face_y2 - face_y1


            left_cheek_x1 = face_x1 + int(0.12 * face_width)
            left_cheek_x2 = face_x1 + int(0.38 * face_width)

            left_cheek_y1 = face_y1 + int(0.48 * face_height)
            left_cheek_y2 = face_y1 + int(0.70 * face_height)

            left_cheek_x1, left_cheek_y1, left_cheek_x2, left_cheek_y2 = clip_coordinates(
                    left_cheek_x1,
                    left_cheek_y1,
                    left_cheek_x2,
                    left_cheek_y2,
                    w,
                    h
                )

            left_cheek_roi = frame[
                            left_cheek_y1:left_cheek_y2,
                            left_cheek_x1:left_cheek_x2
                            ]

            right_cheek_x1 = face_x1 + int(0.62 * face_width)
            right_cheek_x2 = face_x1 + int(0.88 * face_width)

            right_cheek_y1 = face_y1 + int(0.48 * face_height)
            right_cheek_y2 = face_y1 + int(0.70 * face_height)


            # Clip coordinates
           

            right_cheek_x1, right_cheek_y1, right_cheek_x2, right_cheek_y2 =clip_coordinates(
                    right_cheek_x1,
                    right_cheek_y1,
                    right_cheek_x2,
                    right_cheek_y2,
                    w,
                    h
                )


            

            right_cheek_roi = frame[
                right_cheek_y1:right_cheek_y2,
                right_cheek_x1:right_cheek_x2
            ]

            cv2.rectangle(
                frame,
                (left_cheek_x1, left_cheek_y1),
                (left_cheek_x2, left_cheek_y2),
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Left Cheek",
                (left_cheek_x1, max(25, left_cheek_y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.rectangle(
                frame,
                (right_cheek_x1, right_cheek_y1),
                (right_cheek_x2, right_cheek_y2),
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Right Cheek",
                (right_cheek_x1, max(25, right_cheek_y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )



    cv2.imshow(
        "UBFC - Cheek ROI Only",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()

face_detector.close()

cv2.destroyAllWindows()