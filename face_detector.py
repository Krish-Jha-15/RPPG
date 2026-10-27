import cv2
import mediapipe as mp
import os


video_path = "vid.avi"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")


#where the python file are stored 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#model (medipipe is stored)
# .tflite meaning is tensor flow lite model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "blaze_face_short_range.tflite"
)


#Initialize MediaPipe
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

    #height,weight,channels
    h, w, _ = frame.shape

    # BGR → RGB
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


    #Draw a box around a face to identify 
    if results.detections:

        #there could be a multiple faces so we are running a loop 
        for detection in results.detections:

            bbox = detection.bounding_box

            # Original MediaPipe bounding box
            x = bbox.origin_x
            y = bbox.origin_y
            box_width = bbox.width
            box_height = bbox.height


            pad_x = int(0.15 * box_width)
            pad_top = int(0.35 * box_height)
            pad_bottom = int(0.10 * box_height)


            # New expanded coordinates
            # x1 = x - pad_x
            x1 = x 
            y1 = y - pad_top

            # x2 = x + box_width + pad_x
            x2 = x + box_width 
            # y2 = y + box_height + pad_bottom
            y2 = y + box_height 


            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w, x2)
            y2 = min(h, y2)


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Full Face",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    cv2.imshow(
        "UBFC - Full Face Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap.release()
face_detector.close()
cv2.destroyAllWindows()