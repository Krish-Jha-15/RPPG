
import cv2
import mediapipe as mp
import os
import numpy as np


def extract_roi_rgb(vid):

    cap = cv2.VideoCapture(vid)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {vid}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "blaze_face_short_range.tflite")

    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        min_detection_confidence=0.5
    )

    face_detector = FaceDetector.create_from_options(options)

    forehead_values = []
    left_cheek_values = []
    right_cheek_values = []
    frame_numbers = []

    frame_count = 0

    def clip_coordinates(x1, y1, x2, y2, w, h):
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        return x1, y1, x2, y2

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            results = face_detector.detect(mp_image)

            if results.detections:

                bbox = results.detections[0].bounding_box

                x, y = bbox.origin_x, bbox.origin_y
                bw, bh = bbox.width, bbox.height

                face_x1 = int(x - 0.15 * bw)
                face_x2 = int(x + 1.15 * bw)
                face_y1 = int(y - 0.35 * bh)
                face_y2 = int(y + 1.10 * bh)

                face_x1, face_y1, face_x2, face_y2 = clip_coordinates(
                    face_x1, face_y1, face_x2, face_y2, w, h
                )

                fw = face_x2 - face_x1
                fh = face_y2 - face_y1

                # Forehead ROI
                fx1 = face_x1 + int(0.20 * fw)
                fx2 = face_x1 + int(0.80 * fw)
                fy1 = face_y1 + int(0.08 * fh)
                fy2 = face_y1 + int(0.30 * fh)

                # Left cheek ROI
                lx1 = face_x1 + int(0.12 * fw)
                lx2 = face_x1 + int(0.38 * fw)
                ly1 = face_y1 + int(0.48 * fh)
                ly2 = face_y1 + int(0.70 * fh)

                # Right cheek ROI
                rx1 = face_x1 + int(0.62 * fw)
                rx2 = face_x1 + int(0.88 * fw)
                ry1 = face_y1 + int(0.48 * fh)
                ry2 = face_y1 + int(0.70 * fh)

                fx1, fy1, fx2, fy2 = clip_coordinates(
                    fx1, fy1, fx2, fy2, w, h
                )
                lx1, ly1, lx2, ly2 = clip_coordinates(
                    lx1, ly1, lx2, ly2, w, h
                )
                rx1, ry1, rx2, ry2 = clip_coordinates(
                    rx1, ry1, rx2, ry2, w, h
                )

                forehead_roi = frame[fy1:fy2, fx1:fx2]
                left_roi = frame[ly1:ly2, lx1:lx2]
                right_roi = frame[ry1:ry2, rx1:rx2]

                # Ensure all ROIs contain pixels
                if all(roi.size > 0 for roi in
                       [forehead_roi, left_roi, right_roi]):

                    forehead_rgb = cv2.cvtColor(
                        forehead_roi, cv2.COLOR_BGR2RGB
                    )
                    left_rgb = cv2.cvtColor(
                        left_roi, cv2.COLOR_BGR2RGB
                    )
                    right_rgb = cv2.cvtColor(
                        right_roi, cv2.COLOR_BGR2RGB
                    )

                    # Average RGB values for each ROI
                    forehead_mean = np.mean(
                        forehead_rgb, axis=(0, 1)
                    )
                    left_mean = np.mean(
                        left_rgb, axis=(0, 1)
                    )
                    right_mean = np.mean(
                        right_rgb, axis=(0, 1)
                    )

                    forehead_values.append(forehead_mean)
                    left_cheek_values.append(left_mean)
                    right_cheek_values.append(right_mean)
                    frame_numbers.append(frame_count)

                    # Display ROI rectangles
                    cv2.rectangle(
                        frame, (fx1, fy1), (fx2, fy2),
                        (255, 0, 0), 2
                    )
                    cv2.putText(
                        frame, "Forehead", (fx1, max(20, fy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2
                    )

                    cv2.rectangle(
                        frame, (lx1, ly1), (lx2, ly2),
                        (0, 255, 255), 2
                    )
                    cv2.putText(
                        frame, "Left Cheek", (lx1, max(20, ly1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2
                    )

                    cv2.rectangle(
                        frame, (rx1, ry1), (rx2, ry2),
                        (0, 255, 255), 2
                    )
                    cv2.putText(
                        frame, "Right Cheek", (rx1, max(20, ry1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2
                    )

            cv2.imshow("ROI Extraction", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        face_detector.close()
        cv2.destroyAllWindows()

    # Convert lists to NumPy arrays
    forehead_values = np.array(forehead_values)
    left_cheek_values = np.array(left_cheek_values)
    right_cheek_values = np.array(right_cheek_values)

    print("Total frames processed:", frame_count)
    print("Frames with valid ROIs:", len(frame_numbers))
    print("Forehead RGB shape:", forehead_values.shape)
    print("Left cheek RGB shape:", left_cheek_values.shape)
    print("Right cheek RGB shape:", right_cheek_values.shape)

    # Save CSV
    output_path = os.path.join(BASE_DIR, "all_roi_rgb.csv")

    if len(frame_numbers) > 0:

        all_roi_data = np.column_stack((
            frame_numbers,
            forehead_values,
            left_cheek_values,
            right_cheek_values
        ))

        np.savetxt(
            output_path,
            all_roi_data,
            delimiter=",",
            header=(
                "Frame,"
                "Forehead_R,Forehead_G,Forehead_B,"
                "LeftCheek_R,LeftCheek_G,LeftCheek_B,"
                "RightCheek_R,RightCheek_G,RightCheek_B"
            ),
            comments="",
            fmt="%.6f"
        )

        print("CSV saved:", output_path)

    else:
        print("No valid face ROIs detected. CSV not saved.")

    return forehead_values, left_cheek_values, right_cheek_values


# forehead, left_cheek, right_cheek = extract_roi_rgb("vid_5.avi")
